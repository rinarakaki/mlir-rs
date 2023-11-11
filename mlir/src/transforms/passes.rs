/*!
# Pass Entrypoints

- include
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Transforms/Passes.h>
  - <https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Transforms/Passes.td>
*/

/// Creates an instance of the Canonicaliser pass, configured with default
/// settings (which can be overridden by pass options on the command line).
std::unique_ptr<Pass> createCanonicaliserPass();

/// Creates an instance of the Canonicaliser pass with the specified config.
/// `disabledPatterns` is a set of labels used to filter out input patterns with
/// a debug label or debug name in this set. `enabledPatterns` is a set of
/// labels used to filter out input patterns that do not have one of the labels
/// in this set. Debug labels must be set explicitly on patterns or when adding
/// them with `RewritePatternSet::addWithLabel`. Debug names may be empty, but
/// patterns created with `RewritePattern::create` have their default debug name
/// set to their type name.
std::unique_ptr<Pass>
createCanonicaliserPass(const GreedyRewriteConfig &config,
                        ArrayRef<std::string> disabledPatterns = std::nullopt,
                        ArrayRef<std::string> enabledPatterns = std::nullopt);

/// Creates a pass to perform control-flow sinking.
std::unique_ptr<Pass> createControlFlowSinkPass();

/// Creates a pass to perform common sub expression elimination.
std::unique_ptr<Pass> createCSEPass();

/// Creates a pass that generates IR to verify ops at runtime.
std::unique_ptr<Pass> createGenerateRuntimeVerificationPass();

/// Creates a loop invariant code motion pass that hoists loop invariant
/// instructions out of the loop.
std::unique_ptr<Pass> createLoopInvariantCodeMotionPass();

/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> createStripDebugInfoPass();

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module.
std::unique_ptr<Pass> createPrintOpStatsPass(raw_ostream &os = llvm::errs());

/// Creates a pass which prints the list of ops and the number of occurrences in
/// the module with the output format option.
std::unique_ptr<Pass> createPrintOpStatsPass(raw_ostream &os, bool printAsJSON);

/// Creates a pass which inlines calls and callable operations as defined by
/// the CallGraph.
std::unique_ptr<Pass> createInlinerPass();

/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// default inliner pipeline during optimization.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines);

/// Creates an instance of the inliner pass, and use the provided pass managers
/// when optimizing callable operations with names matching the key type.
/// Callable operations with a name not within the provided map will use the
/// provided default pipeline builder.
std::unique_ptr<Pass>
createInlinerPass(llvm::StringMap<OpPassManager> opPipelines,
                  std::function<void(OpPassManager &)> defaultPipelineBuilder);

/// Creates a pass which performs sparse conditional constant propagation over
/// nested operations.
std::unique_ptr<Pass> createSCCPPass();

/// Creates a pass which delete symbol operations that are unreachable. This
/// pass may *only* be scheduled on an operation that defines a SymbolTable.
std::unique_ptr<Pass> createSymbolDCEPass();

/// Creates a pass which marks top-level symbol operations as `private` unless
/// listed in `excludeSymbols`.
std::unique_ptr<Pass>
createSymbolPrivatizePass(ArrayRef<std::string> excludeSymbols = {});

/// Creates a pass that recursively sorts nested regions without SSA dominance
/// topologically such that, as much as possible, users of values appear after
/// their producers.
std::unique_ptr<Pass> createTopologicalSortPass();
