

.pick_ctx_bracket <- function(est, brackets = c(4096,5120,6144,7168,8192,10240,15360,20480,40960), default_ctx) {
  if (is.null(est) || is.na(est) || length(est)==0) return(default_ctx)
  chosen <- brackets[which(brackets >= est)[1]]
  if (is.na(chosen)) max(brackets) else chosen
}


#' Annotate a single text according to coding instructions
#'
#' @description
#' `bacchuss_satyr()` annotates a single text. It's used within `bacchuss()`.
#' It uses the instructions you hand over, and detects whether you added
#' example cases and explanations. Please write instructions that tell the
#' llm to output labels after Labels:
#' If your instructions tell the llm to output an explanation first, and
#' set explanation = TRUE, or hand over a set of explanations for your few shot
#' examples. bacchuss_satyr returns everything after Explanation: as well.
#'
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
#' @param input_text The text to annotate
#' @param expected_response_format Do you use plaintext or json format? Default plain.
#' @param est_ctx_len The estimated context length for instructions, examples,
#' codes and the text to code.
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
#'
#' @returns
#' A list with:
#' \describe{
#'   \item{label}{Annotated label for the text.}
#'   \item{explanation}{Explanation when requested.}
#'   \item{raw_output}{Full LLM output.}
#'   \item{max_context_used}{Context sent to the model.}
#'   \item{tokens_prompt}{Prompt token count.}
#'   \item{tokens_response}{Response token count.}
#' }
#' @export
bacchuss_satyr <- function(instructions, examples = NULL, explanations = NULL, codes = NULL,
                           reminder_s, reminder, input_text, est_ctx_len = NA, expected_response_format = c("plain","json"),
                           model, host = NULL, api_key = NULL, backend = c("ollama","litellm"),
                           temperature = 0.2, seed = NULL,
                           retry_sleep = 30, max_retries = 3, default_ctx = 5120,
                           retry_loop = TRUE) {
  expected_response_format <- match.arg(expected_response_format)
  backend <- match.arg(backend)
  max_ctx <- .pick_ctx_bracket(est_ctx_len, default_ctx = default_ctx)

  ## Errors that explain user mistakes

  if (is.null(examples)) {
    if (!is.null(codes)) {
      stop("`codes` must be NULL when `examples` is NULL (zero-shot mode).", call. = FALSE)
    }
  } else {
    if (is.null(codes)) {
      stop("`codes` must be provided when `examples` are used (few-shot mode).", call. = FALSE)
    }
    if (length(codes) != length(examples)) {
      stop("`codes` and `examples` must have the same length.", call. = FALSE)
    }
    if (is.character(explanations) && length(explanations) != length(examples)) {
      stop("`explanations` must match the length of `examples`.", call. = FALSE)
    }
  }

  if (backend == "litellm" && (is.null(host) || !nzchar(host))) {
    stop("`host` must be set for backend = 'litellm'.", call. = FALSE)
  }


  # normalize examples, explanations and codes
  if (!is.null(examples)) {
    if (length(examples) == 1 && is.na(examples)) examples <- NULL else examples[is.na(examples)] <- ""
  }
  if (!is.null(explanations)) {
    if (length(explanations) == 1 && is.na(explanations)) explanations <- NULL else explanations[is.na(explanations)] <- ""
  }
  if (!is.null(codes)) {
    if (length(codes) == 1 && is.na(codes)) codes <- NULL else codes[is.na(codes)] <- ""
  }

  # generate messages
  system_message <- list(role="system", content=instructions)

  example_messages <- if (is.null(examples)) {
    list()
  } else {
    purrr::map2(seq_along(examples), examples, function(i, ex_text) {
      user_msg <- list(role="user", content = paste0("Text: '", ex_text, "'\nReminder: ", reminder_s))
      if (expected_response_format == "json") {
        out_list <- list()
        if (is.character(explanations) && !is.na(explanations[i]) && nzchar(explanations[i])) {
          out_list$explanation <- explanations[i]
        }
        out_list$label <- codes[i]
        example_content <- jsonlite::toJSON(out_list, auto_unbox = TRUE)
      } else {
        example_content <- paste0(
          ifelse(
            is.character(explanations) && !is.na(explanations[i]) && nzchar(explanations[i]),
            paste0("Explanation: ", explanations[i], "\n"),
            ""
          ),
          "Label: ", codes[i]
        )
      }
      assistant_msg <- list(role = "assistant", content = example_content)
      list(user_msg, assistant_msg)
    }) |> purrr::list_flatten()
  }

  input_message <- list(role="user", content = paste0("Text: '", input_text, "'\nReminder: ", reminder))
  messages <- c(list(system_message), example_messages, list(input_message))
  seed_s <- seed

  response_text <- NULL
  tokens_prompt <- NA
  tokens_response <- NA
  resp <- NULL

  for (i in seq_len(max_retries)) {
    if (backend == "ollama") {
      req <- ollamar::chat(model = model,
                           messages = messages,
                           keep_alive = "10m",
                           output = "req",
                           temperature = temperature,
                           num_ctx = max_ctx,
                           host = host,
                           seed = seed_s)
      if (!is.null(api_key) && nzchar(api_key)) {
        req <- httr2::req_headers(req, "Authorization" = paste0("Bearer ", api_key))
      }
      res <- tryCatch({
        resp <- httr2::req_perform(req)
        ollamar::resp_process(resp, output = "text")
      }, error = function(e) NULL)

      # accept only if token counters present
      # ollama returns NULL token counts when it loops endlessly

      ok_tokens <- tryCatch(!is.null(resp$cache[[ls(resp$cache)[1]]]$prompt_eval_count), error=function(e) FALSE)
      if (!is.null(res) && (ok_tokens || !retry_loop)) {
        response_text <- res
        if (ok_tokens) {
          tokens_prompt <- httr2::resp_body_json(resp)$prompt_eval_count
          tokens_response <- httr2::resp_body_json(resp)$eval_count
        }
        break
      }
    } else {
      url <- paste0(host, "/chat/completions")
      body <- list(
        model = model,
        messages = messages,
        temperature = temperature,
        num_ctx = max_ctx
      )
      if (!is.null(seed_s)) body$seed <- seed_s
      req <- httr2::request(url) |>
        httr2::req_body_json(body)

      if (!is.null(api_key) && nzchar(api_key)) {
        req <- req |> httr2::req_headers(Authorization = paste0("Bearer ", api_key))
      }

      j <- tryCatch({
        resp <- httr2::req_perform(req)
        httr2::resp_body_json(resp)
      }, error = function(e) NULL)

      if (!is.null(j)) {
        response_text <- j$choices[[1]]$message$content
        tokens_prompt <- j$usage$prompt_tokens
        tokens_response <- j$usage$completion_tokens
        break
      }
    }

    if (i < max_retries) {
      message(sprintf("Attempt %d failed, retrying.", i))
      Sys.sleep(retry_sleep)
      seed_s <- if (is.null(seed_s)) NULL else seed_s + 1
    } else {
      stop(sprintf("All %d attempts failed.", max_retries))
    }
  }

  if (expected_response_format == "json") {
    txt_clean <- stringr::str_remove(
      response_text,
      stringr::regex("^```[jJ]?[sS]?[oO]?[nN]?\\s*", dotall = TRUE)
    )
    txt_clean <- stringr::str_remove(
      txt_clean,
      stringr::regex("```\\s*$", dotall = TRUE)
    )

    json_str <- stringr::str_extract(
      txt_clean,
      stringr::regex("\\{.*\\}", dotall = TRUE)
    )

    if (!is.na(json_str)) {
      j <- tryCatch(jsonlite::fromJSON(json_str), error = function(e) NULL)
      label <- if (!is.null(j$label)) as.character(j$label) else NA_character_
      if (!is.null(explanations)) {
        explanation <- if (!is.null(j$explanation)) as.character(j$explanation) else NA_character_
      } else {
        explanation <- NULL
      }
    } else {
      label <- NA_character_
      explanation <- if (!is.null(explanations)) NA_character_ else NULL
    }
  } else {
    label <- stringr::str_match(response_text, "Label:\\s*(.*?)(\\n|$)")[, 2]
    if (!is.null(explanations)) {
      explanation <- stringr::str_match(
        response_text,
        "Explanation:\\s*((?s).*?)(?=\\nLabel:|\\n*$)"
      )[, 2]
    }
  }

  list(
    label = stringr::str_trim(label),
    explanation = if (is.null(explanations)) NULL else stringr::str_trim(explanation),
    raw_output = response_text,
    max_context_used = max_ctx,
    tokens_prompt = tokens_prompt,
    tokens_response = tokens_response
  )
}


#' Annotate texts according to coding instructions
#'
#' @description
#' `bacchuss()` annotates the text in the dataframe you hand over.
#' It uses the instructions you hand over, and detects whether you added
#' example cases and explanations. Please write instructions that tell the
#' llm to output labels after Labels:
#' If your instructions tell the llm to output an explanation first, and
#' set explanation = TRUE, or hand over a set of explanations for your few shot
#' examples. bacchuss returns everything after Explanation: as well.
#' Sometimes, small LLMs will be stuck in short loops. Default behavior is to
#' message "Attempt failed" and then retry.
#'
#' @param df A data frame containing the texts you want to annotate
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
#'
#' @returns
#' A data frame with added columns:
#' \describe{
#'   \item{labels}{Annotated labels.}
#'   \item{explanations}{Explanations when requested.}
#'   \item{raw_output}{Full LLM output per row.}
#'   \item{max_context_used}{Context used.}
#'   \item{tokens_prompt}{Prompt tokens.}
#'   \item{tokens_response}{Response tokens.}
#' }
#' @export
bacchuss <- function(df, input_column = "text", ctx_column = "estimated_context_length",
                    instructions, examples=NULL, explanations=NULL, codes=NULL,
                    reminder_s = "", reminder, expected_response_format = c("plain","json"),
                    model, host = NULL, api_key = NULL, backend = c("ollama","litellm"),
                    temperature = 0.2, seed = NULL,
                    retry_loop = TRUE,
                    retry_sleep = 30, max_retries = 3,
                    default_ctx = 5120, warmup = 0) {

  expected_response_format <- match.arg(expected_response_format)
  backend <- match.arg(backend)

  if (!(input_column %in% names(df))) {
    stop(sprintf("Input column '%s' not found in dataframe.", input_column), call. = FALSE)
  }

  if (!(ctx_column %in% names(df))) {
    message(sprintf("Estimated context length column not found. Using default_ctx = %d.", default_ctx))
  }

  if (nrow(df) == 0) {
    stop("Input dataframe has 0 rows.", call. = FALSE)
  }

  df$labels <- NA_character_
  if (!is.null(explanations)) df$explanations <- NA_character_
  df$raw_output <- NA_character_
  df$max_context_used <- NA_integer_
  df$tokens_prompt <- NA_integer_
  df$tokens_response <- NA_integer_

  has_est <- ctx_column %in% names(df)

  if (warmup > 0) {
    df_warmup <- df[rep(seq_len(nrow(df)), length.out = warmup), ]
    message(sprintf("Running warmup: %d calls.", nrow(df_warmup)))
    pb_warmup <- progress::progress_bar$new(
      format = "  warmup [:bar] :current/:total (:percent) in :elapsed ETA: :eta",
      total = nrow(df_warmup),
      clear = FALSE,
      width = 60
    )
    for (i in seq_len(nrow(df_warmup))) {
      est_ctx_len <- if (has_est) df_warmup[[ctx_column]][i] else NA
      bacchuss_satyr(
        instructions=instructions,
        examples=examples,
        explanations=explanations,
        codes=codes,
        reminder_s=reminder_s,
        reminder=reminder,
        input_text=df_warmup[[input_column]][i],
        est_ctx_len=est_ctx_len,
        model=model,
        temperature=temperature,
        host = host,
        api_key = api_key,
        retry_loop = retry_loop,
        retry_sleep = retry_sleep,
        max_retries = max_retries,
        default_ctx = default_ctx,
        seed = seed,
        backend = backend,
        expected_response_format = expected_response_format
      )
      pb_warmup$tick()
    }
  }

  pb <- progress::progress_bar$new(format="  [:bar] :current/:total (:percent) in :elapsed ETA: :eta",
                                   total = nrow(df), clear = FALSE, width = 60)

  for (i in seq_len(nrow(df))) {
    est_ctx_len <- if (has_est) df[[ctx_column]][i] else NA
    result <- bacchuss_satyr(
      instructions=instructions,
      examples=examples,
      explanations=explanations,
      codes=codes,
      reminder_s=reminder_s,
      reminder=reminder,
      input_text=df[[input_column]][i],
      est_ctx_len=est_ctx_len,
      model=model,
      temperature=temperature,
      host = host,
      api_key = api_key,
      retry_loop = retry_loop,
      retry_sleep = retry_sleep,
      max_retries = max_retries,
      default_ctx = default_ctx,
      seed = seed,
      backend = backend,
      expected_response_format = expected_response_format
    )
    df$labels[i] <- result$label
    if(!is.null(explanations)){
      df$explanations[i] <- result$explanation
    }
    df$raw_output[i] <- result$raw_output
    df$max_context_used[i] <- result$max_context_used
    df$tokens_prompt[i] <- result$tokens_prompt
    df$tokens_response[i] <- result$tokens_response
    pb$tick()
  }
  df
}
