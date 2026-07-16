// Copyright 2026 The Jujutsu Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Storage-neutral access to a position-ordered commit graph.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use itertools::Itertools as _;

use crate::backend::ChangeId;
use crate::backend::CommitId;
use crate::hex_util;
use crate::index::Index;
use crate::index::IndexError;
use crate::index::IndexResult;
use crate::object_id::HexPrefix;
use crate::object_id::ObjectId as _;
use crate::object_id::PrefixResolution;
use crate::repo_path::RepoPathBuf;
use crate::revset::ResolvedExpression;
use crate::revset::Revset;
use crate::revset::RevsetEvaluationError;
use crate::store::Store;

/// Global position of a commit in a position-ordered index.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct GlobalPosition(pub u32);

impl GlobalPosition {
    /// Lowest valid global position.
    pub const MIN: Self = Self(u32::MIN);
    /// Sentinel greater than every valid global position.
    pub const MAX: Self = Self(u32::MAX);
}

/// Owned graph data for one indexed commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexGraphEntry {
    /// Commit identifier.
    pub commit_id: CommitId,
    /// Change identifier stored by the commit.
    pub change_id: ChangeId,
    /// Longest parent generation plus one.
    pub generation_number: u32,
    /// Parent positions, all lower than this entry's position.
    pub parent_positions: Vec<GlobalPosition>,
}

/// Provides fallible, storage-neutral access to a position-ordered commit
/// graph.
///
/// Entries and changed paths are owned so implementations may release or evict
/// backing pages as soon as a method returns. Implementations should preserve
/// the invariant that parents have lower positions than their children.
pub trait PositionIndex: Send + Sync {
    /// Returns the number of indexed commits.
    fn num_commits(&self) -> u32;

    /// Resolves an exact commit ID to its global position.
    fn position_by_commit_id(&self, id: &CommitId) -> IndexResult<Option<GlobalPosition>>;

    /// Loads the graph entry at `position`.
    fn entry_by_position(&self, position: GlobalPosition) -> IndexResult<IndexGraphEntry>;

    /// Resolves a commit-ID prefix across the full index.
    fn resolve_commit_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>>;

    /// Resolves a change-ID prefix to matching commit positions.
    ///
    /// Positions in a single match must be sorted in descending order.
    fn resolve_change_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<Vec<GlobalPosition>>>;

    /// Loads sorted changed paths for the commit at `position`.
    ///
    /// Returns `None` if changed paths were not indexed for this commit.
    fn changed_paths(&self, position: GlobalPosition) -> IndexResult<Option<Vec<RepoPathBuf>>>;
}

impl<T: PositionIndex + ?Sized> PositionIndex for &T {
    fn num_commits(&self) -> u32 {
        T::num_commits(self)
    }

    fn position_by_commit_id(&self, id: &CommitId) -> IndexResult<Option<GlobalPosition>> {
        T::position_by_commit_id(self, id)
    }

    fn entry_by_position(&self, position: GlobalPosition) -> IndexResult<IndexGraphEntry> {
        T::entry_by_position(self, position)
    }

    fn resolve_commit_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>> {
        T::resolve_commit_id_prefix(self, prefix)
    }

    fn resolve_change_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<Vec<GlobalPosition>>> {
        T::resolve_change_id_prefix(self, prefix)
    }

    fn changed_paths(&self, position: GlobalPosition) -> IndexResult<Option<Vec<RepoPathBuf>>> {
        T::changed_paths(self, position)
    }
}

impl<T: PositionIndex + ?Sized> PositionIndex for Arc<T> {
    fn num_commits(&self) -> u32 {
        T::num_commits(self)
    }

    fn position_by_commit_id(&self, id: &CommitId) -> IndexResult<Option<GlobalPosition>> {
        T::position_by_commit_id(self, id)
    }

    fn entry_by_position(&self, position: GlobalPosition) -> IndexResult<IndexGraphEntry> {
        T::entry_by_position(self, position)
    }

    fn resolve_commit_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>> {
        T::resolve_commit_id_prefix(self, prefix)
    }

    fn resolve_change_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<Vec<GlobalPosition>>> {
        T::resolve_change_id_prefix(self, prefix)
    }

    fn changed_paths(&self, position: GlobalPosition) -> IndexResult<Option<Vec<RepoPathBuf>>> {
        T::changed_paths(self, position)
    }
}

/// Evaluates a resolved revset with JJ's default position-based revset engine.
///
/// The returned revset owns a cloneable shared index handle; graph and
/// changed-path read failures are reported lazily as
/// [`RevsetEvaluationError::Index`].
pub fn evaluate_revset(
    expression: &ResolvedExpression,
    store: &Arc<Store>,
    index: Arc<dyn PositionIndex>,
) -> Result<Box<dyn Revset>, RevsetEvaluationError> {
    let revset = crate::default_index::revset_engine::evaluate(expression, store, index)?;
    Ok(Box::new(revset))
}

/// Returns whether `ancestor` is reachable from `descendant`, including
/// equality.
pub fn is_ancestor(
    index: &dyn PositionIndex,
    ancestor: &CommitId,
    descendant: &CommitId,
) -> IndexResult<bool> {
    let ancestor_pos = required_position(index, ancestor)?;
    let descendant_pos = required_position(index, descendant)?;
    is_ancestor_positions(index, ancestor_pos, descendant_pos)
}

/// Position-based implementation of [`is_ancestor()`].
pub fn is_ancestor_positions(
    index: &dyn PositionIndex,
    ancestor_pos: GlobalPosition,
    descendant_pos: GlobalPosition,
) -> IndexResult<bool> {
    let ancestor_generation = index.entry_by_position(ancestor_pos)?.generation_number;
    let mut work = vec![descendant_pos];
    let mut visited = HashSet::new();
    while let Some(position) = work.pop() {
        match position.cmp(&ancestor_pos) {
            Ordering::Less => continue,
            Ordering::Equal => return Ok(true),
            Ordering::Greater => {}
        }
        if !visited.insert(position) {
            continue;
        }
        let entry = index.entry_by_position(position)?;
        if entry.generation_number > ancestor_generation {
            work.extend(entry.parent_positions);
        }
    }
    Ok(false)
}

/// Returns the best common ancestors of two commit sets.
pub fn common_ancestors(
    index: &dyn PositionIndex,
    set1: &[CommitId],
    set2: &[CommitId],
) -> IndexResult<Vec<CommitId>> {
    let positions1 = set1
        .iter()
        .map(|id| required_position(index, id))
        .try_collect()?;
    let positions2 = set2
        .iter()
        .map(|id| required_position(index, id))
        .try_collect()?;
    common_ancestor_positions(index, positions1, positions2)?
        .into_iter()
        .map(|position| Ok(index.entry_by_position(position)?.commit_id))
        .collect()
}

/// Returns best common-ancestor positions in descending order.
pub fn common_ancestor_positions(
    index: &dyn PositionIndex,
    set1: Vec<GlobalPosition>,
    set2: Vec<GlobalPosition>,
) -> IndexResult<Vec<GlobalPosition>> {
    let mut items1 = BinaryHeap::from(set1);
    let mut items2 = BinaryHeap::from(set2);
    let mut result = Vec::new();
    while let (Some(&pos1), Some(&pos2)) = (items1.peek(), items2.peek()) {
        match pos1.cmp(&pos2) {
            Ordering::Greater => shift_to_parents(&mut items1, pos1, index)?,
            Ordering::Less => shift_to_parents(&mut items2, pos2, index)?,
            Ordering::Equal => {
                result.push(pos1);
                dedup_pop(&mut items1);
                dedup_pop(&mut items2);
            }
        }
    }
    heads_positions(index, result)
}

/// Returns commits from `candidates` which are not ancestors of another
/// candidate.
pub fn heads(
    index: &dyn PositionIndex,
    candidates: &mut dyn Iterator<Item = &CommitId>,
) -> IndexResult<Vec<CommitId>> {
    let positions = candidates
        .map(|id| required_position(index, id))
        .try_collect()?;
    heads_positions(index, positions)?
        .into_iter()
        .map(|position| Ok(index.entry_by_position(position)?.commit_id))
        .collect()
}

/// Returns head positions among `candidates`, sorted in descending order.
pub fn heads_positions(
    index: &dyn PositionIndex,
    mut candidates: Vec<GlobalPosition>,
) -> IndexResult<Vec<GlobalPosition>> {
    candidates.sort_unstable_by_key(|&position| Reverse(position));
    candidates.dedup();
    let Some(min_generation) = candidates
        .iter()
        .map(|&position| Ok(index.entry_by_position(position)?.generation_number))
        .collect::<IndexResult<Vec<_>>>()?
        .into_iter()
        .min()
    else {
        return Ok(candidates);
    };

    let mut parents = BinaryHeap::new();
    let mut heads = Vec::new();
    'outer: for candidate in candidates {
        while let Some(&parent) = parents.peek().filter(|&&parent| parent >= candidate) {
            let entry = index.entry_by_position(parent)?;
            if entry.generation_number <= min_generation {
                dedup_pop(&mut parents);
            } else {
                shift_to_parents_with_entry(&mut parents, parent, &entry)?;
            }
            if parent == candidate {
                continue 'outer;
            }
        }
        let entry = index.entry_by_position(candidate)?;
        parents.extend(entry.parent_positions);
        heads.push(candidate);
    }
    Ok(heads)
}

/// Returns positions which are heads among all indexed commits.
pub fn all_head_positions(index: &dyn PositionIndex) -> IndexResult<Vec<GlobalPosition>> {
    let num_commits = index.num_commits();
    let mut not_heads = HashSet::new();
    for position in (0..num_commits).map(GlobalPosition) {
        not_heads.extend(index.entry_by_position(position)?.parent_positions);
    }
    Ok((0..num_commits)
        .map(GlobalPosition)
        .filter(|position| !not_heads.contains(position))
        .collect())
}

/// Returns IDs which are heads among all indexed commits.
pub fn all_heads(index: &dyn PositionIndex) -> IndexResult<Vec<CommitId>> {
    all_head_positions(index)?
        .into_iter()
        .map(|position| Ok(index.entry_by_position(position)?.commit_id))
        .collect()
}

/// Finds heads in the position range `roots..heads` while applying `filter`.
///
/// The returned positions and filter calls are in descending position order.
pub fn heads_from_range_and_filter<E>(
    index: &dyn PositionIndex,
    roots: Vec<GlobalPosition>,
    heads: Vec<GlobalPosition>,
    parents_range: &Range<u32>,
    mut filter: impl FnMut(GlobalPosition) -> Result<bool, E>,
) -> Result<Vec<GlobalPosition>, E>
where
    E: From<IndexError>,
{
    if heads.is_empty() {
        return Ok(heads);
    }
    let mut wanted_queue = BinaryHeap::from(heads);
    let mut unwanted_queue = BinaryHeap::from(roots);
    let mut found_heads = Vec::new();
    while let Some(&position) = wanted_queue.peek() {
        if shift_to_parents_until(&mut unwanted_queue, index, position)? {
            dedup_pop(&mut wanted_queue);
            continue;
        }
        let entry = index.entry_by_position(position)?;
        if filter(position)? {
            dedup_pop(&mut wanted_queue);
            unwanted_queue.extend(entry.parent_positions);
            found_heads.push(position);
        } else {
            let parent_positions = filter_slice_by_range(&entry.parent_positions, parents_range);
            shift_to_parents_from_slice(&mut wanted_queue, position, parent_positions)?;
        }
    }
    Ok(found_heads)
}

fn required_position(
    index: &dyn PositionIndex,
    commit_id: &CommitId,
) -> IndexResult<GlobalPosition> {
    index
        .position_by_commit_id(commit_id)?
        .ok_or_else(|| IndexError::CommitNotFound(commit_id.clone()))
}

fn shift_to_parents_until<E>(
    queue: &mut BinaryHeap<GlobalPosition>,
    index: &dyn PositionIndex,
    target_pos: GlobalPosition,
) -> Result<bool, E>
where
    E: From<IndexError>,
{
    while let Some(&position) = queue.peek().filter(|&&position| position >= target_pos) {
        shift_to_parents(queue, position, index)?;
        if position == target_pos {
            return Ok(true);
        }
    }
    Ok(false)
}

fn shift_to_parents<E>(
    queue: &mut BinaryHeap<GlobalPosition>,
    position: GlobalPosition,
    index: &dyn PositionIndex,
) -> Result<(), E>
where
    E: From<IndexError>,
{
    let entry = index.entry_by_position(position)?;
    shift_to_parents_with_entry(queue, position, &entry)
}

fn shift_to_parents_with_entry<E>(
    queue: &mut BinaryHeap<GlobalPosition>,
    position: GlobalPosition,
    entry: &IndexGraphEntry,
) -> Result<(), E>
where
    E: From<IndexError>,
{
    shift_to_parents_from_slice(queue, position, &entry.parent_positions)
}

fn shift_to_parents_from_slice<E>(
    queue: &mut BinaryHeap<GlobalPosition>,
    position: GlobalPosition,
    parent_positions: &[GlobalPosition],
) -> Result<(), E>
where
    E: From<IndexError>,
{
    let mut parents = parent_positions.iter();
    if let Some(&parent) = parents.next() {
        validate_parent_position(parent, position)?;
        dedup_replace(queue, parent);
    } else {
        dedup_pop(queue);
        return Ok(());
    }
    for &parent in parents {
        validate_parent_position(parent, position)?;
        queue.push(parent);
    }
    Ok(())
}

fn validate_parent_position<E>(parent: GlobalPosition, child: GlobalPosition) -> Result<(), E>
where
    E: From<IndexError>,
{
    if parent < child {
        Ok(())
    } else {
        Err(IndexError::InvalidParentPosition { parent, child }.into())
    }
}

fn dedup_pop<T: Ord>(heap: &mut BinaryHeap<T>) -> Option<T> {
    let item = heap.pop()?;
    remove_dup(heap, &item);
    Some(item)
}

fn dedup_replace<T: Ord>(heap: &mut BinaryHeap<T>, new_item: T) -> Option<T> {
    let old_item = {
        let mut item = heap.peek_mut()?;
        mem::replace(&mut *item, new_item)
    };
    remove_dup(heap, &old_item);
    Some(old_item)
}

fn remove_dup<T: Ord>(heap: &mut BinaryHeap<T>, item: &T) {
    while let Some(entry) = heap.peek_mut().filter(|entry| **entry == *item) {
        std::collections::binary_heap::PeekMut::pop(entry);
    }
}

fn filter_slice_by_range<'a, T>(slice: &'a [T], range: &Range<u32>) -> &'a [T] {
    let start = (range.start as usize).min(slice.len());
    let end = (range.end as usize).min(slice.len());
    &slice[start..end]
}

/// Adapts a [`PositionIndex`] to JJ's higher-level [`Index`] interface.
///
/// This is used internally by tree-diff revset predicates and is also useful to
/// implementations which want the shared graph algorithms without duplicating
/// the `Index` methods.
pub struct PositionIndexAdapter {
    index: Arc<dyn PositionIndex>,
}

impl PositionIndexAdapter {
    /// Creates an adapter owning a shared index handle.
    pub fn new(index: Arc<dyn PositionIndex>) -> Self {
        Self { index }
    }
}

impl Index for PositionIndexAdapter {
    fn shortest_unique_commit_id_prefix_len(&self, commit_id: &CommitId) -> IndexResult<usize> {
        let mut length = 0;
        for position in (0..self.index.num_commits()).map(GlobalPosition) {
            let other_id = self.index.entry_by_position(position)?.commit_id;
            if &other_id != commit_id {
                length = length
                    .max(hex_util::common_hex_len(commit_id.as_bytes(), other_id.as_bytes()) + 1);
            }
        }
        Ok(length)
    }

    fn resolve_commit_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>> {
        self.index.resolve_commit_id_prefix(prefix)
    }

    fn has_id(&self, commit_id: &CommitId) -> IndexResult<bool> {
        Ok(self.index.position_by_commit_id(commit_id)?.is_some())
    }

    fn is_ancestor(&self, ancestor_id: &CommitId, descendant_id: &CommitId) -> IndexResult<bool> {
        is_ancestor(self.index.as_ref(), ancestor_id, descendant_id)
    }

    fn common_ancestors(&self, set1: &[CommitId], set2: &[CommitId]) -> IndexResult<Vec<CommitId>> {
        common_ancestors(self.index.as_ref(), set1, set2)
    }

    fn all_heads_for_gc(&self) -> IndexResult<Box<dyn Iterator<Item = CommitId> + '_>> {
        Ok(Box::new(all_heads(self.index.as_ref())?.into_iter()))
    }

    fn heads(&self, candidates: &mut dyn Iterator<Item = &CommitId>) -> IndexResult<Vec<CommitId>> {
        heads(self.index.as_ref(), candidates)
    }

    fn changed_paths_in_commit(
        &self,
        commit_id: &CommitId,
    ) -> IndexResult<Option<Box<dyn Iterator<Item = RepoPathBuf> + '_>>> {
        let Some(position) = self.index.position_by_commit_id(commit_id)? else {
            return Ok(None);
        };
        Ok(self
            .index
            .changed_paths(position)?
            .map(|paths| Box::new(paths.into_iter()) as Box<dyn Iterator<Item = RepoPathBuf>>))
    }

    fn evaluate_revset(
        &self,
        expression: &ResolvedExpression,
        store: &Arc<Store>,
    ) -> Result<Box<dyn Revset + '_>, RevsetEvaluationError> {
        evaluate_revset(expression, store, self.index.clone())
    }
}
