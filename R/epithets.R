#' Annotate texts - test disagreement to find hard cases for your annotation instructions
#'
#' @description
#' `briseus()` annotates the text in the dataframe you hand over. It runs annotation
#' n_runs times. You can then inspect in which cases the annotation disagreed - those
#' are cases that you might need additional instructions for.
#' It uses the instructions you hand over, and detects whether you added
#' example cases and explanations. Please write instructions that tell the
#' llm to output labels after Labels:
#' If your instructions tell the llm to output an explanation first, and
#' set explanation = TRUE, or hand over a set of explanations for your few shot
#' examples. isodaetes returns everything after Explanation: as well.
#' Sometimes, small LLMs will be stuck in short loops. Default behavior is to
#' message "Attempt failed" and then retry.
#'
#' @param df A data frame containing the texts you want to annotate
#' @param n_runs Number of annotation runs over the same dataset. Default: 3.
#' @param input_column A char indicating the column the texts to code
#' are stored. Default: "text"
#' @param ctx_column A char indicating where the context length for
#' the call is stored. Leave at default if there is none.
#' Default: "estimated_context_length". If that column doesn't exist:
#' Throws a message that it uses default_ctx, then uses default_ctx.
#' Can go up to 40960 if your llm allows it.
#' @param instructions A char with your coding instructions
#' @param examples,codes Two vectors with annotation examples: the
#' example text and the correct label for each case
#' @param explanations Leave Null if you don't want explanations for
#' each label.
#'    * If you use zero shot and want explanations, set TRUE
#'    * If you use few shot, give a vector with explanations for the
#'    example labels
#' @param reminder_s Short reminders to send after each coding example.
#' Can be left empty, does not usually improve reliability and costs
#' too many tokens
#' @param reminder Short reminder sent after all coding examples. Can
#' Increase stability of labels - repeat instructions for available labels,
#' rules the model needs reminder of (for example "do not interpret, only
#' go by what's explicit in the text").
#' @param expected_response_format Do you use plaintext or json format? Default plain.
#' @param model Which LLM model to use. No default.
#' @param host Address of the host you use. Default: local ollama
#' installation (uses default from ollamar package)
#' @param api_key Which API key to use (if your host handes users with
#' API keys). Default: NULL for unrestricted APIs.
#' @param backend Are you using a local Ollama instance or a LiteLLM host?
#' @param temperature Which temperature to use for the LLM. Higher =
#' more creative, lower = more consistent. Anything above 1 is not advised.
#' Default set to 0.2.
#' @param seed Seed to use for deterministic annotation. In unsuccessful
#' tries, will try again with seed + 1. Default: NULL
#' @param retry_loop Small LLMs sometimes get into a long loop producing irrelevant
#' text. If set TRUE, the results of this run will be discarded and the LLM will
#' rerun with seed + 1. If set FALSE, keep result from loop - you can inspect the
#' raw output to check what happened.
#' @param retry_sleep How long to wait before retrying - to avoid over-
#' loading local ollama instances. Default: 30 seconds.
#' @param max_retries Max number of retries per call. Usually only needs
#' one retry. Default: 3.
#' @param default_ctx Default context length if no context length is handed
#' over. Default: 5120.
#' @param warmup Ollama sometimes acts non-deterministically even if you set a seed.
#' to adjust to that you can hand over a warmup-value - the first cases will be run
#' in a warm-up loop before the system runs the actual cases. Default: 0
#' @param max_tokens Cap length of response - forces the llm to stop generating 
#' when max token length is reached. Use when llm fails and you suspect infinite
#' generation of text is the cause.
#' @param reasoning Whether to request reasoning from the model.
#'   One of FALSE, TRUE, NULL, "low", "medium", or "high".
#'   Defaults to FALSE.
#'   For Ollama, this is sent as `think`.
#'   For LiteLLM, FALSE maps to `reasoning_effort = "none"`,
#'   TRUE maps to `"medium"`, and "low"/"medium"/"high" are passed through.
#'   Reasoning behavior is model- and backend-dependent and not guaranteed.
#'   Use reasoning = NULL to omit the parameter if needed.
#'
#' @returns
#' A data frame with added columns:
#' \describe{
#'   \item{labels_runx}{Annotated labels by run number.}
#'   \item{explanations_runx}{Explanations when requested by run number.}
#'   \item{majority_label}{Label that got annotated in most runs}
#'   \item{majority_count}{How often majority label was chosen.}
#'   \item{agreement}{majority_count / number of runs}
#' }
#' @export
briseus <- function(df,
                    n_runs = 3,
                    input_column = "text",
                    ctx_column = "estimated_context_length",
                    instructions, examples = NULL, explanations = NULL, codes = NULL,
                    reminder_s = "", reminder,
                    expected_response_format = c("plain", "json"),
                    model, host = NULL, api_key = NULL, backend = c("ollama", "litellm"),
                    temperature = 0.2, seed = NULL,
                    retry_loop = TRUE,
                    retry_sleep = 30, max_retries = 3,
                    default_ctx = 5120, warmup = 0,
                    max_tokens = NULL,
                    reasoning = FALSE) {
  
  expected_response_format <- match.arg(expected_response_format)
  backend <- match.arg(backend)
  
  if (!is.data.frame(df)) stop("`df` must be a data frame.", call. = FALSE)
  if (!is.numeric(n_runs) || length(n_runs) != 1 || is.na(n_runs) || n_runs < 1) {
    stop("`n_runs` must be a single integer >= 1.", call. = FALSE)
  }
  
  .validate_reasoning(reasoning)
  n_runs <- as.integer(n_runs)
  
  # store run outputs
  labels_mat <- matrix(NA_character_, nrow = nrow(df), ncol = n_runs)
  expl_mat <- if (!is.null(explanations)) matrix(NA_character_, nrow = nrow(df), ncol = n_runs) else NULL
  
  for (r in seq_len(n_runs)) {
    message(sprintf("briseus: starting run %d/%d", r, n_runs))
    
    # make runs reproducible but different, if seed is provided
    seed_r <- if (is.null(seed)) NULL else seed + (r - 1)
    
    res_r <- bacchuss(
      df = df,
      input_column = input_column,
      ctx_column = ctx_column,
      instructions = instructions,
      examples = examples,
      explanations = explanations,
      codes = codes,
      reminder_s = reminder_s,
      reminder = reminder,
      expected_response_format = expected_response_format,
      model = model,
      host = host,
      api_key = api_key,
      backend = backend,
      temperature = temperature,
      seed = seed_r,
      retry_loop = retry_loop,
      retry_sleep = retry_sleep,
      max_retries = max_retries,
      default_ctx = default_ctx,
      warmup = warmup,
      max_tokens = max_tokens,
      reasoning = reasoning
    )
    
    labels_mat[, r] <- res_r$labels
    if (!is.null(expl_mat)) expl_mat[, r] <- res_r$explanations
  }
  
  # add per-run columns
  out <- df
  for (r in seq_len(n_runs)) {
    out[[sprintf("labels_run%d", r)]] <- labels_mat[, r]
    if (!is.null(expl_mat)) {
      out[[sprintf("explanations_run%d", r)]] <- expl_mat[, r]
    }
  }
  
  # majority label + agreement per row
  majority_label <- character(nrow(out))
  majority_count <- integer(nrow(out))
  agreement <- numeric(nrow(out))
  
  for (i in seq_len(nrow(out))) {
    labs <- labels_mat[i, ]
    labs <- labs[!is.na(labs) & nzchar(labs)]
    
    if (length(labs) == 0) {
      majority_label[i] <- NA_character_
      majority_count[i] <- 0L
      agreement[i] <- NA_real_
      next
    }
    
    tab <- table(labs)
    max_n <- max(tab)
    
    # tie-break: pick first label in run order among tied labels
    tied <- names(tab)[tab == max_n]
    pick <- labs[which(labs %in% tied)[1]]
    
    majority_label[i] <- pick
    majority_count[i] <- max_n
    agreement[i] <- max_n / n_runs
  }
  
  out$majority_label <- majority_label
  out$majority_count <- majority_count
  out$agreement <- agreement
  
  out
}


#' Annotate texts - majority out of x according to coding instructions, jury for ties
#'
#' @description
#' `isodaetes()` annotates the text in the dataframe you hand over.
#' It uses the instructions you hand over, and detects whether you added
#' example cases and explanations. Please write instructions that tell the
#' llm to output labels after Labels:
#' If your instructions tell the llm to output an explanation first, and
#' set explanation = TRUE, or hand over a set of explanations for your few shot
#' examples. isodaetes returns everything after Explanation: as well.
#' Sometimes, small LLMs will be stuck in short loops. Default behavior is to
#' message "Attempt failed" and then retry.
#'
#' @param df A data frame containing the texts you want to annotate
#' @param n_runs Number of annotation runs over the same dataset. Default: 3.
#' @param agreement_threshold Agreement threshold - if majority vote is below the 
#' threshold, llm jury is applied. Default: 0.5
#' @param input_column A char indicating the column the texts to code
#' are stored. Default: "text"
#' @param ctx_column A char indicating where the context length for
#' the call is stored. Leave at default if there is none.
#' Default: "estimated_context_length". If that column doesn't exist:
#' Throws a message that it uses default_ctx, then uses default_ctx.
#' Can go up to 40960 if your llm allows it.
#' @param instructions A char with your coding instructions
#' @param examples,codes Two vectors with annotation examples: the
#' example text and the correct label for each case
#' @param explanations Leave Null if you don't want explanations for
#' each label.
#'    * If you use zero shot and want explanations, set TRUE
#'    * If you use few shot, give a vector with explanations for the
#'    example labels
#' @param reminder_s Short reminders to send after each coding example.
#' Can be left empty, does not usually improve reliability and costs
#' too many tokens
#' @param reminder Short reminder sent after all coding examples. Can
#' Increase stability of labels - repeat instructions for available labels,
#' rules the model needs reminder of (for example "do not interpret, only
#' go by what's explicit in the text").
#' @param expected_response_format Do you use plaintext or json format? Default plain.
#' @param model Which LLM model to use. No default.
#' @param host Address of the host you use. Default: local ollama
#' installation (uses default from ollamar package)
#' @param api_key Which API key to use (if your host handes users with
#' API keys). Default: NULL for unrestricted APIs.
#' @param backend Are you using a local Ollama instance or a LiteLLM host?
#' @param temperature Which temperature to use for the LLM. Higher =
#' more creative, lower = more consistent. Anything above 1 is not advised.
#' Default set to 0.2.
#' @param seed Seed to use for deterministic annotation. In unsuccessful
#' tries, will try again with seed + 1. Default: NULL
#' @param retry_loop Small LLMs sometimes get into a long loop producing irrelevant
#' text. If set TRUE, the results of this run will be discarded and the LLM will
#' rerun with seed + 1. If set FALSE, keep result from loop - you can inspect the
#' raw output to check what happened.
#' @param retry_sleep How long to wait before retrying - to avoid over-
#' loading local ollama instances. Default: 30 seconds.
#' @param max_retries Max number of retries per call. Usually only needs
#' one retry. Default: 3.
#' @param default_ctx Default context length if no context length is handed
#' over. Default: 5120.
#' @param warmup Ollama sometimes acts non-deterministically even if you set a seed.
#' to adjust to that you can hand over a warmup-value - the first cases will be run
#' in a warm-up loop before the system runs the actual cases. Default: 0
#' @param jury_append_prompt Prompt that will be appended to the jury call to explain jury function. 
#' If NULL uses english default.
#' @param jury_temperature Temperature for the jury call. Default: 0
#' @param jury_seed Seed for the jury call. Default: NULL
#' @param jury_reminder Reminder to append to jury call - default is in English.
#' @param max_tokens Cap length of response - forces the llm to stop generating 
#' when max token length is reached. Use when llm fails and you suspect infinite
#' generation of text is the cause.
#' @param reasoning Whether to request reasoning from the model.
#'   One of FALSE, TRUE, NULL, "low", "medium", or "high".
#'   Defaults to FALSE.
#'   For Ollama, this is sent as `think`.
#'   For LiteLLM, FALSE maps to `reasoning_effort = "none"`,
#'   TRUE maps to `"medium"`, and "low"/"medium"/"high" are passed through.
#'   Reasoning behavior is model- and backend-dependent and not guaranteed.
#'   Use reasoning = NULL to omit the parameter if needed.
#'
#' @returns
#' A data frame with added columns:
#' \describe{
#'   \item{labels_runx}{Annotated labels by run number.}
#'   \item{explanations_runx}{Explanations when requested by run number.}
#'   \item{majority_label}{Label that got annotated in most runs}
#'   \item{majority_count}{How often majority label was chosen.}
#'   \item{agreement}{majority_count / number of runs}
#'   \item{final_label}{Chosen label}
#'   \item{final_explanation}{Final explanation when requested.}
#'   \item{used_jury}{Whether jury call was used.}
#' }
#' @export
isodaetes <- function(df,
                      n_runs = 3,
                      agreement_threshold = 0.5,
                      input_column = "text",
                      ctx_column = "estimated_context_length",
                      instructions, examples = NULL, explanations = NULL, codes = NULL,
                      reminder_s = "", reminder,
                      expected_response_format = c("plain", "json"),
                      model, host = NULL, api_key = NULL, backend = c("ollama", "litellm"),
                      temperature = 0.2, seed = NULL,
                      retry_loop = TRUE,
                      retry_sleep = 30, max_retries = 3,
                      default_ctx = 5120, warmup = 0,
                      jury_append_prompt = NULL,
                      jury_temperature = 0.0,
                      jury_seed = NULL,
                      jury_reminder = "Decide the best single label based strictly on the coding instructions. Output only in the required format.",
                      max_tokens = NULL,
                      reasoning = FALSE) {
  
  expected_response_format <- match.arg(expected_response_format)
  backend <- match.arg(backend)
  
  if (!is.data.frame(df)) stop("`df` must be a data frame.", call. = FALSE)
  if (!is.numeric(n_runs) || length(n_runs) != 1 || is.na(n_runs) || n_runs < 1) {
    stop("`n_runs` must be a single integer >= 1.", call. = FALSE)
  }
  n_runs <- as.integer(n_runs)
  
  if (!is.numeric(agreement_threshold) || length(agreement_threshold) != 1 ||
      is.na(agreement_threshold) || agreement_threshold <= 0 || agreement_threshold > 1) {
    stop("`agreement_threshold` must be a single number in (0, 1].", call. = FALSE)
  }
  
  .validate_reasoning(reasoning)
  
  if (is.null(jury_append_prompt)) {
    jury_append_prompt <- paste0(
      "\n\nJURY TASK:\n",
      "You are acting as a jury/arbitrator. You will receive: (1) the Text, ",
      "(2) the proposed labels from multiple independent codings, and optionally their explanations. ",
      "Choose the single best label according to the coding instructions. ",
      "Do not average; pick exactly one label from the allowed set."
    )
  }
  
  jury_instructions <- paste0(instructions, jury_append_prompt)
  
  labels_mat <- matrix(NA_character_, nrow = nrow(df), ncol = n_runs)
  expl_mat <- if (!is.null(explanations)) matrix(NA_character_, nrow = nrow(df), ncol = n_runs) else NULL
  
  for (r in seq_len(n_runs)) {
    message(sprintf("isodaetes: starting run %d/%d", r, n_runs))
    seed_r <- if (is.null(seed)) NULL else seed + (r - 1)
    
    res_r <- bacchuss(
      df = df,
      input_column = input_column,
      ctx_column = ctx_column,
      instructions = instructions,
      examples = examples,
      explanations = explanations,
      codes = codes,
      reminder_s = reminder_s,
      reminder = reminder,
      expected_response_format = expected_response_format,
      model = model,
      host = host,
      api_key = api_key,
      backend = backend,
      temperature = temperature,
      seed = seed_r,
      retry_loop = retry_loop,
      retry_sleep = retry_sleep,
      max_retries = max_retries,
      default_ctx = default_ctx,
      warmup = warmup,
      max_tokens = max_tokens,
      reasoning = reasoning
    )
    
    labels_mat[, r] <- res_r$labels
    if (!is.null(expl_mat)) expl_mat[, r] <- res_r$explanations
  }
  
  out <- df
  for (r in seq_len(n_runs)) {
    out[[sprintf("labels_run%d", r)]] <- labels_mat[, r]
    if (!is.null(expl_mat)) out[[sprintf("explanations_run%d", r)]] <- expl_mat[, r]
  }
  
  majority_label <- character(nrow(out))
  majority_count <- integer(nrow(out))
  agreement <- numeric(nrow(out))
  
  for (i in seq_len(nrow(out))) {
    labs <- labels_mat[i, ]
    labs <- labs[!is.na(labs) & nzchar(labs)]
    
    if (length(labs) == 0) {
      majority_label[i] <- NA_character_
      majority_count[i] <- 0L
      agreement[i] <- NA_real_
      next
    }
    
    tab <- table(labs)
    max_n <- max(tab)
    
    tied <- names(tab)[tab == max_n]
    pick <- labs[which(labs %in% tied)[1]]
    
    majority_label[i] <- pick
    majority_count[i] <- max_n
    agreement[i] <- max_n / n_runs
  }
  
  out$majority_label <- majority_label
  out$majority_count <- majority_count
  out$agreement <- agreement
  
  # IMPORTANT CHANGE:
  # jury if agreement is BELOW threshold
  needs_jury <- which(!is.na(out$agreement) & out$agreement < agreement_threshold)
  
  out$final_label <- out$majority_label
  if (!is.null(explanations)) out$final_explanation <- out$explanations_run1 else out$final_explanation <- NULL
  out$used_jury <- FALSE
  
  if (length(needs_jury) > 0) {
    message(sprintf("isodaetes: running jury for %d/%d rows (agreement < %.2f)",
                    length(needs_jury), nrow(out), agreement_threshold))
    
    for (k in seq_along(needs_jury)) {
      i <- needs_jury[k]
      message(sprintf("  jury row %d/%d (df row %d)", k, length(needs_jury), i))
      
      txt <- out[[input_column]][i]
      
      proposed_labels <- paste0("Run ", seq_len(n_runs), ": ", labels_mat[i, ], collapse = "\n")
      
      jury_text <- paste0(
        "Text: '", txt, "'\n",
        "Proposed labels:\n", proposed_labels, "\n"
      )
      
      if (!is.null(expl_mat)) {
        proposed_expl <- paste0("Run ", seq_len(n_runs), ": ", expl_mat[i, ], collapse = "\n")
        jury_text <- paste0(jury_text, "Proposed explanations:\n", proposed_expl, "\n")
      }
      
      est_ctx_len <- if (ctx_column %in% names(out)) out[[ctx_column]][i] else NA
      
      jury_res <- bacchuss_satyr(
        instructions = jury_instructions,
        examples = examples,
        explanations = explanations,
        codes = codes,
        reminder_s = reminder_s,
        reminder = jury_reminder,
        input_text = jury_text,
        est_ctx_len = est_ctx_len,
        model = model,
        host = host,
        api_key = api_key,
        backend = backend,
        temperature = jury_temperature,
        seed = jury_seed,
        retry_loop = retry_loop,
        retry_sleep = retry_sleep,
        max_retries = max_retries,
        default_ctx = default_ctx,
        expected_response_format = expected_response_format,
        max_tokens = max_tokens,
        reasoning = reasoning
      )
      
      out$final_label[i] <- jury_res$label
      if (!is.null(explanations)) out$final_explanation[i] <- jury_res$explanation
      out$used_jury[i] <- TRUE
    }
  }
  
  out
}

#' Test few-shot examples with leave-one-out annotation
#'
#' @description
#' `liknites()` is a function to test few-shot annotation examples. Using the leave-one-out
#' method, we test how well the instruction plus all few-shot examples except one would code
#' the left-out example. 
#' The output will tell you which examples are informative (by having low agreement) and which
#' ones are more redundant.
#'
#' @param instructions A char with your coding instructions
#' @param examples,codes Two vectors with annotation examples: the
#' example text and the correct label for each case
#' @param explanations Leave Null if you don't want explanations for
#' each label.
#'    * If you use zero shot and want explanations, set TRUE
#'    * If you use few shot, give a vector with explanations for the
#'    example labels
#' @param n_runs Number of annotation runs over the same examples. Default: 3.
#' @param reminder_s Short reminders to send after each coding example.
#' Can be left empty, does not usually improve reliability and costs
#' too many tokens
#' @param reminder Short reminder sent after all coding examples. Can
#' Increase stability of labels - repeat instructions for available labels,
#' rules the model needs reminder of (for example "do not interpret, only
#' go by what's explicit in the text").
#' @param expected_response_format Do you use plaintext or json format? Default plain.
#' @param model Which LLM model to use. No default.
#' @param host Address of the host you use. Default: local ollama
#' installation (uses default from ollamar package)
#' @param api_key Which API key to use (if your host handes users with
#' API keys). Default: NULL for unrestricted APIs.
#' @param backend Are you using a local Ollama instance or a LiteLLM host?
#' @param temperature Which temperature to use for the LLM. Higher =
#' more creative, lower = more consistent. Anything above 1 is not advised.
#' Default set to 0.2.
#' @param seed Seed to use for deterministic annotation. In unsuccessful
#' tries, will try again with seed + 1. Default: NULL
#' @param retry_loop Small LLMs sometimes get into a long loop producing irrelevant
#' text. If set TRUE, the results of this run will be discarded and the LLM will
#' rerun with seed + 1. If set FALSE, keep result from loop - you can inspect the
#' raw output to check what happened.
#' @param retry_sleep How long to wait before retrying - to avoid over-
#' loading local ollama instances. Default: 30 seconds.
#' @param max_retries Max number of retries per call. Usually only needs
#' one retry. Default: 3.
#' @param default_ctx Default context length if no context length is handed
#' over. Default: 5120.
#' @param verbose If update messages are sent. Default: TRUE.
#' @param max_tokens Cap length of response - forces the llm to stop generating 
#' when max token length is reached. Use when llm fails and you suspect infinite
#' generation of text is the cause.
#' @param reasoning Whether to request reasoning from the model.
#'   One of FALSE, TRUE, NULL, "low", "medium", or "high".
#'   Defaults to FALSE.
#'   For Ollama, this is sent as `think`.
#'   For LiteLLM, FALSE maps to `reasoning_effort = "none"`,
#'   TRUE maps to `"medium"`, and "low"/"medium"/"high" are passed through.
#'   Reasoning behavior is model- and backend-dependent and not guaranteed.
#'   Use reasoning = NULL to omit the parameter if needed.
#'
#' @returns
#' A data frame with added columns:
#' \describe{
#' 
#'   
#'   \item{example_id}{An id for each example}
#'   \item{text}{The example text.}
#'   \item{true_label}{The original label.}
#'   \item{labels_runx}{Annotated labels by run number.}
#'   \item{explanations_runx}{Explanations when requested by run number.}
#'   \item{majority_label}{Label that got annotated in most runs}
#'   \item{majority_count}{How often majority label was chosen.}
#'   \item{agreement}{majority_count / number of runs}
#'   \item{correct_majority}{Whether the majority call was correct}
#' }
#' @export
liknites <- function(instructions,
                                  examples, codes,
                                  explanations = NULL,
                                  n_runs = 3,
                                  reminder_s = "",
                                  reminder,
                                  expected_response_format = c("plain", "json"),
                                  model, host = NULL, api_key = NULL, backend = c("ollama", "litellm"),
                                  temperature = 0.2, seed = NULL,
                                  retry_loop = TRUE,
                                  retry_sleep = 30, max_retries = 3,
                                  default_ctx = 5120,
                                  verbose = TRUE,
                     max_tokens = NULL,
                     reasoning = FALSE) {
  
  expected_response_format <- match.arg(expected_response_format)
  backend <- match.arg(backend)
  
  if (is.null(examples) || is.null(codes)) stop("`examples` and `codes` must be provided.", call. = FALSE)
  if (length(examples) != length(codes)) stop("`examples` and `codes` must have the same length.", call. = FALSE)
  if (length(examples) < 2) stop("Need at least 2 examples for leave-one-out testing.", call. = FALSE)
  
  if (!is.null(explanations) && is.character(explanations) && length(explanations) != length(examples)) {
    stop("If provided, `explanations` must have the same length as `examples`.", call. = FALSE)
  }
  
  if (!is.numeric(n_runs) || length(n_runs) != 1 || is.na(n_runs) || n_runs < 1) {
    stop("`n_runs` must be a single integer >= 1.", call. = FALSE)
  }
  
  .validate_reasoning(reasoning)
  n_runs <- as.integer(n_runs)
  
  # normalize NAs like your main code does
  examples <- if (length(examples) == 1 && is.na(examples)) NULL else { examples[is.na(examples)] <- ""; examples }
  codes <- if (length(codes) == 1 && is.na(codes)) NULL else { codes[is.na(codes)] <- ""; codes }
  if (!is.null(explanations)) {
    explanations <- if (length(explanations) == 1 && is.na(explanations)) NULL else { explanations[is.na(explanations)] <- ""; explanations }
  }
  
  n_ex <- length(examples)
  
  pred_labels <- matrix(NA_character_, nrow = n_ex, ncol = n_runs)
  pred_expl <- if (!is.null(explanations)) matrix(NA_character_, nrow = n_ex, ncol = n_runs) else NULL
  
  total_calls <- n_ex * n_runs
  
  pb <- progress::progress_bar$new(
    format = "  testing few-shot [:bar] :current/:total (:percent) in :elapsed ETA: :eta",
    total = total_calls,
    clear = FALSE,
    width = 70
  )
  
  if (verbose) {
    message(sprintf("liknites: %d examples, %d runs each (%d total calls).",
                    n_ex, n_runs, total_calls))
  }
  
  call_counter <- 0
  
  for (j in seq_len(n_ex)) {
    

    
    target_text <- examples[j]
    
    idx <- setdiff(seq_len(n_ex), j)
    shots_examples <- examples[idx]
    shots_codes <- codes[idx]
    shots_expl <- if (!is.null(explanations)) explanations[idx] else NULL
    
    for (r in seq_len(n_runs)) {
      call_counter <- call_counter + 1
      

      
      seed_r <- if (is.null(seed)) NULL else seed + (j - 1) * n_runs + (r - 1)
      
      res <- bacchuss_satyr(
        instructions = instructions,
        examples = shots_examples,
        explanations = shots_expl,
        codes = shots_codes,
        reminder_s = reminder_s,
        reminder = reminder,
        input_text = target_text,
        est_ctx_len = NA,
        expected_response_format = expected_response_format,
        model = model,
        host = host,
        api_key = api_key,
        backend = backend,
        temperature = temperature,
        seed = seed_r,
        retry_loop = retry_loop,
        retry_sleep = retry_sleep,
        max_retries = max_retries,
        default_ctx = default_ctx,
        max_tokens = max_tokens,
        reasoning = reasoning
      )
      
      pred_labels[j, r] <- res$label
      if (!is.null(pred_expl)) pred_expl[j, r] <- res$explanation
      
      pb$tick()
    }
  }
  
  # summarize per example
  majority_label <- character(n_ex)
  majority_count <- integer(n_ex)
  agreement <- numeric(n_ex)
  
  for (j in seq_len(n_ex)) {
    labs <- pred_labels[j, ]
    labs <- labs[!is.na(labs) & nzchar(labs)]
    
    if (length(labs) == 0) {
      majority_label[j] <- NA_character_
      majority_count[j] <- 0L
      agreement[j] <- NA_real_
      next
    }
    
    tab <- table(labs)
    max_n <- max(tab)
    tied <- names(tab)[tab == max_n]
    pick <- labs[which(labs %in% tied)[1]]
    
    majority_label[j] <- pick
    majority_count[j] <- max_n
    agreement[j] <- max_n / n_runs
  }
  
  out <- data.frame(
    example_id = seq_len(n_ex),
    text = examples,
    true_label = as.character(codes),
    majority_pred = majority_label,
    majority_count = majority_count,
    agreement = agreement,
    correct_majority = ifelse(is.na(majority_label), NA, majority_label == as.character(codes)),
    stringsAsFactors = FALSE
  )
  
  for (r in seq_len(n_runs)) {
    out[[sprintf("pred_label_run%d", r)]] <- pred_labels[, r]
    if (!is.null(pred_expl)) out[[sprintf("pred_expl_run%d", r)]] <- pred_expl[, r]
  }
  
  out
}

