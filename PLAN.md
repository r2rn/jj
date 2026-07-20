# Plan: Make index sub-traits async

## Context
- The target scope is the sub-index traits in `lib/src/index.rs`: `Index`, `ReadonlyIndex`, `MutableIndex` query/getter methods, and `ChangeIdIndex`.
- `IndexStore::get_index_at_op()` and `MutableIndex::add_commit()` are already async. `Index` query methods (`has_id`, `is_ancestor`, `common_ancestors`, `heads`, `changed_paths_in_commit`, `evaluate_revset`, etc.) and `ChangeIdIndex` query methods are currently synchronous.
- `ReadonlyIndex` provides getters/factories (`as_index()`, `change_id_index()`, `start_modification()`); `MutableIndex` provides getters plus mutating methods (`add_commit()`, `merge_in()`). The requested `MutableIndex` scope is only getter/query paths, not additional mutation methods.
- Initial scan found many affected call sites in `lib/src/repo.rs`, `lib/src/refs.rs`, `lib/src/revset.rs`, `lib/src/id_prefix.rs`, `lib/src/rewrite.rs`, `lib/src/git.rs`, `lib/src/git_backend.rs`, `lib/src/commit.rs`, `cli/src/commit_templater.rs`, CLI commands/benchmarks, and index tests.

## Approach
- Use the existing `async_trait` crate concretely: add `#[async_trait(?Send)]` above `Index`, `ReadonlyIndex`, `MutableIndex`, and `ChangeIdIndex` in `lib/src/index.rs`, change selected trait methods from `fn` to `async fn`, and add matching `#[async_trait(?Send)] impl ...` blocks for each implementation. This preserves `dyn Index`/`dyn ReadonlyIndex`/`dyn MutableIndex`/`dyn ChangeIdIndex` object usage by having `async_trait` box the futures behind the trait object.
- Prefer `?Send` for these index traits because several methods return borrowed iterators/revsets tied to `&self`; this matches the existing `IndexStore` style and avoids over-constraining returned futures.
- Keep default-index logic behaviorally unchanged: default implementations will generally become `async fn` wrappers around existing in-memory/composite operations.
- Propagate async outward through callers that perform index queries, converting synchronous helper APIs to async where needed.
- Avoid changing `MutableIndex` mutation semantics beyond what is already async (`add_commit()`); keep `merge_in()` synchronous unless compilation or borrowing constraints force a narrower adjustment.

## Files to modify
- Core traits: `lib/src/index.rs`
- Default implementations: `lib/src/default_index/composite.rs`, `lib/src/default_index/readonly.rs`, `lib/src/default_index/mutable.rs`, `lib/src/default_index/store.rs`
- Main call-site ripple: `lib/src/repo.rs`, `lib/src/refs.rs`, `lib/src/revset.rs`, `lib/src/id_prefix.rs`, `lib/src/rewrite.rs`, `lib/src/git.rs`, `lib/src/git_backend.rs`, `lib/src/backend.rs`, `lib/src/store.rs`, `lib/src/commit.rs`, `lib/src/commit_builder.rs`, `lib/src/converge.rs`
- CLI/benchmark call sites that directly query indexes, e.g. `cli/src/commands/bench/*.rs`, `cli/src/commands/bookmark/mod.rs`, `cli/src/commands/git/push.rs`, `cli/src/commands/duplicate.rs`, `cli/src/commands/gerrit/upload.rs`, `cli/src/commands/rebase.rs`, `cli/src/commit_templater.rs`
- Tests: `lib/tests/test_index.rs`, `lib/tests/test_operations.rs`, `lib/tests/test_mut_repo.rs`, `lib/tests/test_git_backend.rs`, `lib/tests/test_git.rs`, and default-index unit tests in `lib/src/default_index/mod.rs`

## Reuse
- Existing `async_trait` dependency and patterns in `lib/src/index.rs`, `lib/src/default_index/store.rs`, backend/op-store traits.
- Existing async propagation patterns using `.await?` and TODO error mappings in `lib/src/repo.rs` and `lib/src/git.rs`.
- Existing `pollster::FutureExt::block_on()` pattern in synchronous adapters such as `cli/src/commit_templater.rs` and `lib/src/store.rs`, useful where broader async conversion is disproportionate.
- Existing `DefaultReadonlyIndex::has_id_impl()` for synchronous internal checks inside index-building code where no trait object async boundary is needed.
- Existing composite/index implementation code in `lib/src/default_index/composite.rs` can stay as the synchronous core behind async trait methods.

## Steps
- [ ] In `lib/src/index.rs`, add `#[async_trait(?Send)]` to `Index` and rewrite its query method signatures as `async fn`, e.g. `async fn has_id(&self, ...) -> IndexResult<bool>` and `async fn evaluate_revset(&self, ...) -> Result<Box<dyn Revset + '_>, ...>`.
- [ ] Add `#[async_trait(?Send)]` to `ChangeIdIndex` and convert `resolve_prefix()` and `shortest_unique_prefix_len()` to `async fn`.
- [ ] Add `#[async_trait(?Send)]` to `ReadonlyIndex`; convert `change_id_index()` and `start_modification()` to `async fn` if they need to cross the async sub-index boundary, while keeping `as_index()` synchronous if possible to avoid unnecessary lifetime/borrow churn.
- [ ] Keep `MutableIndex` under async-trait; convert only getter/query entry points (`as_index()`, `change_id_index()` as needed) while leaving `merge_in()` synchronous and preserving already-async `add_commit()`.
- [ ] Update `DefaultReadonlyIndex`, `DefaultMutableIndex`, `CompositeIndex`, and `ChangeIdIndexImpl` trait impls to match; delegate to existing synchronous internals.
- [ ] Make `Repo` change-id helper methods async (`resolve_change_id()`, `resolve_change_id_prefix()`, `shortest_unique_change_id_prefix_len()`) using `async_trait(?Send)`, and preserve `ReadonlyRepo` caching behavior for `change_id_index` where possible.
- [ ] Update repo/view/ref/revset/id-prefix/git/backend APIs that call index queries so they become async where necessary and await index results.
- [ ] Convert backend/store GC to async if needed because `Backend::gc()` receives `&dyn Index` and calls `all_heads_for_gc()`.
- [ ] Update CLI command call sites, commit template/property helpers, and benchmark routines to await async index calls; for synchronous template properties, use the existing `block_on()` adapter pattern unless the local API is already async.
- [ ] Update tests and test helpers to async or add awaits where index queries are made.

## Verification
- `cargo fmt`
- `cargo test -p jj-lib test_index`
- `cargo test -p jj-lib test_mut_repo test_operations test_git_backend`
- Run a broader `cargo test -p jj-lib` because the async ripple touches repo/revset/git/change-id behavior extensively.
- Compile/test affected CLI code, especially commit templates and benchmark commands.
