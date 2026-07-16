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

//! Storage-neutral records produced by a mutable commit index.
//!
//! Unlike the default index segment format, these records refer to parents by
//! commit ID. A storage implementation can therefore append a delta against a
//! newer base index and assign its own final positions.

use crate::backend::ChangeId;
use crate::backend::CommitId;
use crate::repo_path::RepoPathBuf;

/// Logical records added by one mutable-index overlay.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IndexDelta {
    /// New commits in parent-before-child topological order.
    pub commits: Vec<IndexCommitRecord>,
    /// Changed paths for commits for which the optional path index was built.
    ///
    /// No record means that changed paths are unavailable, while a record with
    /// an empty `paths` vector means the commit is known to change no paths.
    pub changed_paths: Vec<ChangedPathRecord>,
}

impl IndexDelta {
    /// Returns true if the delta contains no graph or changed-path records.
    pub fn is_empty(&self) -> bool {
        self.commits.is_empty() && self.changed_paths.is_empty()
    }
}

/// A commit graph record independent of persisted index positions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IndexCommitRecord {
    /// Content-addressed commit identifier.
    pub commit_id: CommitId,
    /// Change identifier carried by the commit.
    pub change_id: ChangeId,
    /// Parent commit identifiers.
    pub parent_ids: Vec<CommitId>,
}

/// Optional changed-path data associated with a commit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChangedPathRecord {
    /// Commit whose changed paths were indexed.
    pub commit_id: CommitId,
    /// Sorted repository paths changed by the commit.
    pub paths: Vec<RepoPathBuf>,
}
