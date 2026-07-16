// Copyright 2023 The Jujutsu Authors
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

#[cfg(test)]
use std::collections::BinaryHeap;
#[cfg(test)]
use std::collections::binary_heap;
use std::iter;
#[cfg(test)]
use std::mem;
#[cfg(test)]
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use itertools::Itertools as _;
use ref_cast::RefCastCustom;
use ref_cast::ref_cast_custom;

use super::bit_set::AncestorsBitSet;
use super::bit_set::PositionsBitSet;
use super::changed_path::CompositeChangedPathIndex;
use super::entry::CommitIndexEntry;
use super::entry::GlobalCommitPosition;
use super::entry::LocalCommitPosition;
use super::entry::SmallGlobalCommitPositionsVec;
use super::entry::SmallLocalCommitPositionsVec;
use super::mutable::MutableCommitIndexSegment;
use super::readonly::ReadonlyCommitIndexSegment;
#[cfg(test)]
use super::rev_walk::filter_slice_by_range;
use crate::backend::ChangeId;
use crate::backend::CommitId;
use crate::hex_util;
use crate::index::ChangeIdIndex;
use crate::index::Index;
use crate::index::IndexError;
use crate::index::IndexResult;
use crate::index::ResolvedChangeState;
use crate::index::ResolvedChangeTargets;
use crate::object_id::HexPrefix;
use crate::object_id::ObjectId as _;
use crate::object_id::PrefixResolution;
use crate::object_id::id_type;
use crate::position_index;
use crate::position_index::IndexGraphEntry;
use crate::position_index::PositionIndex;
use crate::repo_path::RepoPathBuf;
use crate::revset::ResolvedExpression;
use crate::revset::Revset;
use crate::revset::RevsetEvaluationError;
use crate::store::Store;

id_type!(pub(super) CommitIndexSegmentId { hex() });

pub(super) trait CommitIndexSegment: Send + Sync {
    fn num_parent_commits(&self) -> u32;

    fn num_local_commits(&self) -> u32;

    fn parent_file(&self) -> Option<&Arc<ReadonlyCommitIndexSegment>>;

    fn commit_id_to_pos(&self, commit_id: &CommitId) -> Option<LocalCommitPosition>;

    /// Suppose the given `commit_id` exists, returns the previous and next
    /// commit ids in lexicographical order.
    fn resolve_neighbor_commit_ids(
        &self,
        commit_id: &CommitId,
    ) -> (Option<CommitId>, Option<CommitId>);

    fn resolve_commit_id_prefix(&self, prefix: &HexPrefix) -> PrefixResolution<CommitId>;

    fn resolve_neighbor_change_ids(
        &self,
        change_id: &ChangeId,
    ) -> (Option<ChangeId>, Option<ChangeId>);

    // Returns positions in ascending order.
    fn resolve_change_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> PrefixResolution<(ChangeId, SmallLocalCommitPositionsVec)>;

    fn generation_number(&self, local_pos: LocalCommitPosition) -> u32;

    fn commit_id(&self, local_pos: LocalCommitPosition) -> CommitId;

    fn change_id(&self, local_pos: LocalCommitPosition) -> ChangeId;

    fn num_parents(&self, local_pos: LocalCommitPosition) -> u32;

    fn parent_positions(&self, local_pos: LocalCommitPosition) -> SmallGlobalCommitPositionsVec;
}

pub(super) type DynCommitIndexSegment = dyn CommitIndexSegment;

/// Abstraction over owned and borrowed types that can be cheaply converted to
/// a `CompositeIndex` reference.
pub(super) trait AsCompositeIndex {
    /// Returns reference wrapper that provides global access to this index.
    fn as_composite(&self) -> &CompositeIndex;
}

impl<T: AsCompositeIndex + ?Sized> AsCompositeIndex for &T {
    fn as_composite(&self) -> &CompositeIndex {
        <T as AsCompositeIndex>::as_composite(self)
    }
}

impl<T: AsCompositeIndex + ?Sized> AsCompositeIndex for &mut T {
    fn as_composite(&self) -> &CompositeIndex {
        <T as AsCompositeIndex>::as_composite(self)
    }
}

/// Provides an index of both commit IDs and change IDs.
///
/// We refer to this as a composite index because it's a composite of multiple
/// nested index segments where each parent segment is roughly twice as large
/// its child. segment. This provides a good balance between read and write
/// performance.
#[derive(RefCastCustom)]
#[repr(transparent)]
pub(super) struct CompositeCommitIndex(DynCommitIndexSegment);

impl CompositeCommitIndex {
    #[ref_cast_custom]
    pub(super) const fn new(segment: &DynCommitIndexSegment) -> &Self;

    /// Iterates parent and its ancestor readonly index segments.
    pub(super) fn ancestor_files_without_local(
        &self,
    ) -> impl Iterator<Item = &Arc<ReadonlyCommitIndexSegment>> {
        let parent_file = self.0.parent_file();
        iter::successors(parent_file, |file| file.parent_file())
    }

    /// Iterates self and its ancestor index segments.
    pub(super) fn ancestor_index_segments(&self) -> impl Iterator<Item = &DynCommitIndexSegment> {
        iter::once(&self.0).chain(
            self.ancestor_files_without_local()
                .map(|file| file.as_ref() as &DynCommitIndexSegment),
        )
    }

    pub fn num_commits(&self) -> u32 {
        self.0.num_parent_commits() + self.0.num_local_commits()
    }

    pub fn has_id(&self, commit_id: &CommitId) -> bool {
        self.commit_id_to_pos(commit_id).is_some()
    }

    pub fn entry_by_pos(&self, pos: GlobalCommitPosition) -> CommitIndexEntry<'_> {
        self.try_entry_by_pos(pos).unwrap()
    }

    fn try_entry_by_pos(&self, pos: GlobalCommitPosition) -> Option<CommitIndexEntry<'_>> {
        self.ancestor_index_segments().find_map(|segment| {
            u32::checked_sub(pos.0, segment.num_parent_commits())
                .map(LocalCommitPosition)
                .filter(|local_pos| local_pos.0 < segment.num_local_commits())
                .map(|local_pos| CommitIndexEntry::new(segment, pos, local_pos))
        })
    }

    pub fn entry_by_id(&self, commit_id: &CommitId) -> Option<CommitIndexEntry<'_>> {
        self.ancestor_index_segments().find_map(|segment| {
            let local_pos = segment.commit_id_to_pos(commit_id)?;
            let pos = GlobalCommitPosition(local_pos.0 + segment.num_parent_commits());
            Some(CommitIndexEntry::new(segment, pos, local_pos))
        })
    }

    pub fn commit_id_to_pos(&self, commit_id: &CommitId) -> Option<GlobalCommitPosition> {
        self.ancestor_index_segments().find_map(|segment| {
            let LocalCommitPosition(local_pos) = segment.commit_id_to_pos(commit_id)?;
            let pos = GlobalCommitPosition(local_pos + segment.num_parent_commits());
            Some(pos)
        })
    }

    pub fn resolve_commit_id_prefix(&self, prefix: &HexPrefix) -> PrefixResolution<CommitId> {
        self.ancestor_index_segments()
            .fold(PrefixResolution::NoMatch, |acc_match, segment| {
                if acc_match == PrefixResolution::AmbiguousMatch {
                    acc_match // avoid checking the parent file(s)
                } else {
                    let local_match = segment.resolve_commit_id_prefix(prefix);
                    acc_match.plus(&local_match)
                }
            })
    }

    /// Suppose the given `commit_id` exists, returns the minimum prefix length
    /// to disambiguate it. The length to be returned is a number of hexadecimal
    /// digits.
    ///
    /// If the given `commit_id` doesn't exist, this will return the prefix
    /// length that never matches with any commit ids.
    pub(super) fn shortest_unique_commit_id_prefix_len(&self, commit_id: &CommitId) -> usize {
        let (prev_id, next_id) = self.resolve_neighbor_commit_ids(commit_id);
        itertools::chain(prev_id, next_id)
            .map(|id| hex_util::common_hex_len(commit_id.as_bytes(), id.as_bytes()) + 1)
            .max()
            .unwrap_or(0)
    }

    /// Suppose the given `commit_id` exists, returns the previous and next
    /// commit ids in lexicographical order.
    pub(super) fn resolve_neighbor_commit_ids(
        &self,
        commit_id: &CommitId,
    ) -> (Option<CommitId>, Option<CommitId>) {
        self.ancestor_index_segments()
            .map(|segment| segment.resolve_neighbor_commit_ids(commit_id))
            .reduce(|(acc_prev_id, acc_next_id), (prev_id, next_id)| {
                (
                    acc_prev_id.into_iter().chain(prev_id).max(),
                    acc_next_id.into_iter().chain(next_id).min(),
                )
            })
            .unwrap()
    }

    /// Suppose the given `change_id` exists, returns the minimum prefix length
    /// to disambiguate it within all the indexed ids including hidden ones.
    pub(super) fn shortest_unique_change_id_prefix_len(&self, change_id: &ChangeId) -> usize {
        let (prev_id, next_id) = self.resolve_neighbor_change_ids(change_id);
        itertools::chain(prev_id, next_id)
            .map(|id| hex_util::common_hex_len(change_id.as_bytes(), id.as_bytes()) + 1)
            .max()
            .unwrap_or(0)
    }

    /// Suppose the given `change_id` exists, returns the previous and next
    /// change ids in lexicographical order. The returned change ids may be
    /// hidden.
    pub(super) fn resolve_neighbor_change_ids(
        &self,
        change_id: &ChangeId,
    ) -> (Option<ChangeId>, Option<ChangeId>) {
        self.ancestor_index_segments()
            .map(|segment| segment.resolve_neighbor_change_ids(change_id))
            .reduce(|(acc_prev_id, acc_next_id), (prev_id, next_id)| {
                (
                    acc_prev_id.into_iter().chain(prev_id).max(),
                    acc_next_id.into_iter().chain(next_id).min(),
                )
            })
            .unwrap()
    }

    /// Resolves the given change id `prefix` to the associated entries. The
    /// returned entries may be hidden.
    ///
    /// The returned index positions are sorted in descending order.
    pub(super) fn resolve_change_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> PrefixResolution<(ChangeId, SmallGlobalCommitPositionsVec)> {
        use PrefixResolution::*;
        self.ancestor_index_segments()
            .fold(NoMatch, |acc_match, segment| {
                if acc_match == AmbiguousMatch {
                    return acc_match; // avoid checking the parent file(s)
                }
                let to_global_pos = {
                    let num_parent_commits = segment.num_parent_commits();
                    move |LocalCommitPosition(pos)| GlobalCommitPosition(pos + num_parent_commits)
                };
                // Similar to PrefixResolution::plus(), but merges matches of the same id.
                match (acc_match, segment.resolve_change_id_prefix(prefix)) {
                    (NoMatch, local_match) => local_match.map(|(id, positions)| {
                        (id, positions.into_iter().rev().map(to_global_pos).collect())
                    }),
                    (acc_match, NoMatch) => acc_match,
                    (AmbiguousMatch, _) => AmbiguousMatch,
                    (_, AmbiguousMatch) => AmbiguousMatch,
                    (SingleMatch((id1, _)), SingleMatch((id2, _))) if id1 != id2 => AmbiguousMatch,
                    (SingleMatch((id, mut acc_positions)), SingleMatch((_, local_positions))) => {
                        acc_positions.extend(local_positions.into_iter().rev().map(to_global_pos));
                        SingleMatch((id, acc_positions))
                    }
                }
            })
    }

    pub(super) fn all_heads_pos(&self) -> impl Iterator<Item = GlobalCommitPosition> + use<> {
        let num_commits = self.num_commits();
        let mut not_head = PositionsBitSet::with_capacity(num_commits);
        for pos in (0..num_commits).map(GlobalCommitPosition) {
            let entry = self.entry_by_pos(pos);
            for parent_pos in entry.parent_positions() {
                not_head.set(parent_pos);
            }
        }
        (0..num_commits)
            .map(GlobalCommitPosition)
            // TODO: can be optimized to use leading/trailing_ones()
            .filter(move |&pos| !not_head.get(pos))
    }

    /// Find the heads of a range of positions `roots..heads`, applying a filter
    /// to the commits in the range. The heads are sorted in descending order.
    /// The filter will also be called in descending index position order.
    #[cfg(test)]
    pub fn heads_from_range_and_filter<E>(
        &self,
        roots: Vec<GlobalCommitPosition>,
        heads: Vec<GlobalCommitPosition>,
        parents_range: &Range<u32>,
        mut filter: impl FnMut(GlobalCommitPosition) -> Result<bool, E>,
    ) -> Result<Vec<GlobalCommitPosition>, E> {
        if heads.is_empty() {
            return Ok(heads);
        }
        let mut wanted_queue = BinaryHeap::from(heads);
        let mut unwanted_queue = BinaryHeap::from(roots);
        let mut found_heads = Vec::new();
        while let Some(&pos) = wanted_queue.peek() {
            if shift_to_parents_until(&mut unwanted_queue, self, pos) {
                dedup_pop(&mut wanted_queue);
                continue;
            }
            let entry = self.entry_by_pos(pos);
            if filter(pos)? {
                dedup_pop(&mut wanted_queue);
                unwanted_queue.extend(entry.parent_positions());
                found_heads.push(pos);
            } else {
                let parent_positions = entry.parent_positions();
                shift_to_parents(
                    &mut wanted_queue,
                    pos,
                    filter_slice_by_range(&parent_positions, parents_range),
                );
            }
        }
        Ok(found_heads)
    }
}

#[derive(Clone, Debug)]
enum CompositeCommitIndexSegment {
    Readonly(Arc<ReadonlyCommitIndexSegment>),
    Mutable(Box<MutableCommitIndexSegment>),
}

#[derive(Clone, Debug)]
pub(super) struct CompositeIndex {
    commits: CompositeCommitIndexSegment,
    changed_paths: CompositeChangedPathIndex,
}

impl CompositeIndex {
    pub(super) fn from_readonly(
        commits: Arc<ReadonlyCommitIndexSegment>,
        changed_paths: CompositeChangedPathIndex,
    ) -> Self {
        Self {
            commits: CompositeCommitIndexSegment::Readonly(commits),
            changed_paths,
        }
    }

    pub(super) fn from_mutable(
        commits: Box<MutableCommitIndexSegment>,
        changed_paths: CompositeChangedPathIndex,
    ) -> Self {
        Self {
            commits: CompositeCommitIndexSegment::Mutable(commits),
            changed_paths,
        }
    }

    pub(super) fn into_mutable(
        self,
    ) -> Option<(Box<MutableCommitIndexSegment>, CompositeChangedPathIndex)> {
        let commits = match self.commits {
            CompositeCommitIndexSegment::Readonly(_) => return None,
            CompositeCommitIndexSegment::Mutable(segment) => segment,
        };
        Some((commits, self.changed_paths))
    }

    pub(super) fn commits(&self) -> &CompositeCommitIndex {
        match &self.commits {
            CompositeCommitIndexSegment::Readonly(segment) => segment.as_composite(),
            CompositeCommitIndexSegment::Mutable(segment) => segment.as_composite(),
        }
    }

    pub(super) fn readonly_commits(&self) -> Option<&Arc<ReadonlyCommitIndexSegment>> {
        match &self.commits {
            CompositeCommitIndexSegment::Readonly(segment) => Some(segment),
            CompositeCommitIndexSegment::Mutable(_) => None,
        }
    }

    pub(super) fn mutable_commits(&mut self) -> Option<&mut MutableCommitIndexSegment> {
        match &mut self.commits {
            CompositeCommitIndexSegment::Readonly(_) => None,
            CompositeCommitIndexSegment::Mutable(segment) => Some(segment),
        }
    }

    pub(super) fn changed_paths(&self) -> &CompositeChangedPathIndex {
        &self.changed_paths
    }

    pub(super) fn changed_paths_mut(&mut self) -> &mut CompositeChangedPathIndex {
        &mut self.changed_paths
    }

    fn resolve_change_targets_for_positions(
        &self,
        positions: &[GlobalCommitPosition],
        reachable_set: &mut AncestorsBitSet,
    ) -> IndexResult<ResolvedChangeTargets> {
        debug_assert!(positions.is_sorted_by(|a, b| a > b));
        let Some(&oldest_position) = positions.last() else {
            return Err(IndexError::Corrupt(
                "change ID lookup has no commit positions".to_owned(),
            ));
        };
        reachable_set.visit_until(self, oldest_position)?;
        let targets = positions
            .iter()
            .map(|&position| -> IndexResult<_> {
                let commit_id = self.entry_by_position(position)?.commit_id;
                let state = if reachable_set.contains(position) {
                    ResolvedChangeState::Visible
                } else {
                    ResolvedChangeState::Hidden
                };
                Ok((commit_id, state))
            })
            .try_collect()?;
        Ok(ResolvedChangeTargets { targets })
    }
}

impl AsCompositeIndex for CompositeIndex {
    fn as_composite(&self) -> &CompositeIndex {
        self
    }
}

impl PositionIndex for CompositeIndex {
    fn num_commits(&self) -> u32 {
        self.commits().num_commits()
    }

    fn position_by_commit_id(&self, id: &CommitId) -> IndexResult<Option<GlobalCommitPosition>> {
        Ok(self.commits().commit_id_to_pos(id))
    }

    fn entry_by_position(&self, position: GlobalCommitPosition) -> IndexResult<IndexGraphEntry> {
        let entry = self
            .commits()
            .try_entry_by_pos(position)
            .ok_or(IndexError::InvalidPosition(position))?;
        Ok(IndexGraphEntry {
            commit_id: entry.commit_id(),
            change_id: entry.change_id(),
            generation_number: entry.generation_number(),
            parent_positions: entry.parent_positions().into_vec(),
        })
    }

    fn resolve_commit_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>> {
        Ok(self.commits().resolve_commit_id_prefix(prefix))
    }

    fn resolve_change_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<Vec<GlobalCommitPosition>>> {
        Ok(self
            .commits()
            .resolve_change_id_prefix(prefix)
            .map(|(_change_id, positions)| positions.into_vec()))
    }

    fn changed_paths(
        &self,
        position: GlobalCommitPosition,
    ) -> IndexResult<Option<Vec<RepoPathBuf>>> {
        if self.commits().try_entry_by_pos(position).is_none() {
            return Err(IndexError::InvalidPosition(position));
        }
        Ok(self
            .changed_paths()
            .changed_paths(position)
            .map(|paths| paths.map(|path| path.to_owned()).collect()))
    }
}

// In revset engine, we need to convert &CompositeIndex to &dyn Index.
impl Index for CompositeIndex {
    fn shortest_unique_commit_id_prefix_len(&self, commit_id: &CommitId) -> IndexResult<usize> {
        Ok(self
            .commits()
            .shortest_unique_commit_id_prefix_len(commit_id))
    }

    fn resolve_commit_id_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<CommitId>> {
        PositionIndex::resolve_commit_id_prefix(self, prefix)
    }

    fn has_id(&self, commit_id: &CommitId) -> IndexResult<bool> {
        Ok(self.position_by_commit_id(commit_id)?.is_some())
    }

    fn is_ancestor(&self, ancestor_id: &CommitId, descendant_id: &CommitId) -> IndexResult<bool> {
        position_index::is_ancestor(self, ancestor_id, descendant_id)
    }

    fn common_ancestors(&self, set1: &[CommitId], set2: &[CommitId]) -> IndexResult<Vec<CommitId>> {
        position_index::common_ancestors(self, set1, set2)
    }

    fn all_heads_for_gc(&self) -> IndexResult<Box<dyn Iterator<Item = CommitId> + '_>> {
        Ok(Box::new(position_index::all_heads(self)?.into_iter()))
    }

    fn heads(
        &self,
        candidate_ids: &mut dyn Iterator<Item = &CommitId>,
    ) -> IndexResult<Vec<CommitId>> {
        position_index::heads(self, candidate_ids)
    }

    fn changed_paths_in_commit(
        &self,
        commit_id: &CommitId,
    ) -> IndexResult<Option<Box<dyn Iterator<Item = RepoPathBuf> + '_>>> {
        let Some(position) = self.position_by_commit_id(commit_id)? else {
            return Ok(None);
        };
        Ok(PositionIndex::changed_paths(self, position)?
            .map(|paths| Box::new(paths.into_iter()) as Box<dyn Iterator<Item = RepoPathBuf>>))
    }

    fn evaluate_revset(
        &self,
        expression: &ResolvedExpression,
        store: &Arc<Store>,
    ) -> Result<Box<dyn Revset + '_>, RevsetEvaluationError> {
        position_index::evaluate_revset(expression, store, Arc::new(self.clone()))
    }
}

pub(super) struct ChangeIdIndexImpl<I> {
    index: I,
    reachable_set: Mutex<AncestorsBitSet>,
}

impl<I: AsCompositeIndex> ChangeIdIndexImpl<I> {
    pub fn new(index: I, heads: &mut dyn Iterator<Item = &CommitId>) -> Self {
        let composite = index.as_composite().commits();
        let mut reachable_set = AncestorsBitSet::with_capacity(composite.num_commits());
        for id in heads {
            reachable_set.add_head(composite.commit_id_to_pos(id).unwrap());
        }
        Self {
            index,
            reachable_set: Mutex::new(reachable_set),
        }
    }
}

impl<I: AsCompositeIndex + Send + Sync> ChangeIdIndex for ChangeIdIndexImpl<I> {
    // Resolves change ID prefix among all IDs.
    //
    // If `SingleMatch` is returned, there is at least one commit with the given
    // change ID (either visible or hidden). `AmbiguousMatch` may be returned even
    // if the prefix is unique within the visible entries.
    fn resolve_prefix(
        &self,
        prefix: &HexPrefix,
    ) -> IndexResult<PrefixResolution<ResolvedChangeTargets>> {
        let index = self.index.as_composite();
        match index.commits().resolve_change_id_prefix(prefix) {
            PrefixResolution::NoMatch => Ok(PrefixResolution::NoMatch),
            PrefixResolution::AmbiguousMatch => Ok(PrefixResolution::AmbiguousMatch),
            PrefixResolution::SingleMatch((_change_id, positions)) => {
                let mut reachable_set = self.reachable_set.lock().unwrap();
                let targets =
                    index.resolve_change_targets_for_positions(&positions, &mut reachable_set)?;
                Ok(PrefixResolution::SingleMatch(targets))
            }
        }
    }

    // Calculates the shortest prefix length of the given `change_id` among all
    // IDs, including hidden entries.
    //
    // The returned length is usually a few digits longer than the minimum
    // length necessary to disambiguate within the visible entries since hidden
    // entries are also considered when determining the prefix length.
    fn shortest_unique_prefix_len(&self, change_id: &ChangeId) -> IndexResult<usize> {
        let index = self.index.as_composite().commits();
        Ok(index.shortest_unique_change_id_prefix_len(change_id))
    }
}

/// Repeatedly `shift_to_parents` until reaching a target position. Returns true
/// if the target position matched a position in the queue.
#[cfg(test)]
fn shift_to_parents_until(
    queue: &mut BinaryHeap<GlobalCommitPosition>,
    index: &CompositeCommitIndex,
    target_pos: GlobalCommitPosition,
) -> bool {
    while let Some(&pos) = queue.peek().filter(|&&pos| pos >= target_pos) {
        shift_to_parents(queue, pos, &index.entry_by_pos(pos).parent_positions());
        if pos == target_pos {
            return true;
        }
    }
    false
}

/// Removes an entry from the queue and replace it with its parents.
#[cfg(test)]
fn shift_to_parents(
    items: &mut BinaryHeap<GlobalCommitPosition>,
    pos: GlobalCommitPosition,
    parent_positions: &[GlobalCommitPosition],
) {
    let mut parent_positions = parent_positions.iter();
    if let Some(&parent_pos) = parent_positions.next() {
        assert!(parent_pos < pos);
        dedup_replace(items, parent_pos).unwrap();
    } else {
        dedup_pop(items).unwrap();
        return;
    }
    for &parent_pos in parent_positions {
        assert!(parent_pos < pos);
        items.push(parent_pos);
    }
}

/// Removes the greatest items (including duplicates) from the heap, returns
/// one.
#[cfg(test)]
fn dedup_pop<T: Ord>(heap: &mut BinaryHeap<T>) -> Option<T> {
    let item = heap.pop()?;
    remove_dup(heap, &item);
    Some(item)
}

/// Removes the greatest items (including duplicates) from the heap, inserts
/// lesser `new_item` to the heap, returns the removed one.
///
/// This is faster than calling `dedup_pop(heap)` and `heap.push(new_item)`
/// especially when `new_item` is the next greatest item.
#[cfg(test)]
fn dedup_replace<T: Ord>(heap: &mut BinaryHeap<T>, new_item: T) -> Option<T> {
    let old_item = {
        let mut x = heap.peek_mut()?;
        mem::replace(&mut *x, new_item)
    };
    remove_dup(heap, &old_item);
    Some(old_item)
}

#[cfg(test)]
fn remove_dup<T: Ord>(heap: &mut BinaryHeap<T>, item: &T) {
    while let Some(x) = heap.peek_mut().filter(|x| **x == *item) {
        binary_heap::PeekMut::pop(x);
    }
}
