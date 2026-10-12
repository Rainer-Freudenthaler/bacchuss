
#' Define open annotation fields
#'
#' @description
#' These functions define open-ended annotation fields for
#' use with [kantharos()].
#'
#' Unlike character vectors, which restrict an annotation
#' to predefined labels, open fields allow the LLM to
#' generate values of a specified data type.
#'
#' @returns
#' An object of class `bac_field` describing the field.
#'
#' @examples
#' bac_open_text()
#' bac_open_number()
#' bac_open_integer()
#' bac_open_boolean()
#'
#' @name bac_open_fields
NULL


#' @rdname bac_open_fields
#' @export
bac_open_text <- function() {
  structure(list(type = "string"), class = "bac_field")
}


#' @rdname bac_open_fields
#' @export
bac_open_number <- function() {
  structure(list(type = "number"), class = "bac_field")
}


#' @rdname bac_open_fields
#' @export
bac_open_integer <- function() {
  structure(list(type = "integer"), class = "bac_field")
}


#' @rdname bac_open_fields
#' @export
bac_open_boolean <- function() {
  structure(list(type = "boolean"), class = "bac_field")
}



#' Construct a JSON Schema for LLM annotations
#'
#' @description
#' `kantharos()` creates a JSON Schema that can be passed to
#' other `bacchuss` functions to control the format of LLM
#' responses.
#'
#' A JSON Schema describes the structure of a JSON object,
#' including which fields it contains, their data types,
#' and their allowed values. When used for structured LLM
#' output, the schema constrains the model to produce
#' responses that follow this structure.
#'
#' For example, if a category allows only `"blue"` and
#' `"red"`, the LLM must choose one of these labels rather
#' than generate an arbitrary response.
#'
#' To define annotation categories, supply a named character
#' vector for each category. For categories with a fixed
#' set of labels, provide all permitted values.
#'
#' For open-ended fields, use [bac_open_text()],
#' [bac_open_number()], [bac_open_integer()], or
#' [bac_open_boolean()].
#'
#' The `schema_format` argument specifies whether the schema
#' is formatted for direct Ollama requests or for a LiteLLM
#' backend using the OpenAI-compatible API.
#'
#' @param ... Named annotation categories. Each category
#'   must be a non-empty character vector of allowed labels
#'   or a `bac_open_*()` field specification.
#' @param schema_format Output format: `"ollama"` (default)
#'   or `"litellm"`.
#' @param json_name Name of the JSON Schema for LiteLLM
#'   backends. Defaults to `"annotation"`. Ignored with a
#'   warning if explicitly supplied for Ollama.
#'
#' @returns
#' An R list containing the JSON Schema for Ollama, or
#' an OpenAI-compatible `response_format` specification
#' for LiteLLM.
#'
#' @examples
#' # Ollama schema
#' schema <- kantharos(
#'   explanation = bac_open_text(),
#'   label = c("positive", "neutral", "negative")
#' )
#'
#' # LiteLLM schema
#' schema <- kantharos(
#'   explanation = bac_open_text(),
#'   label = c("positive", "negative"),
#'   schema_format = "litellm",
#'   json_name = "sentiment_annotation"
#' )
#'
#' @export
kantharos <- function(
    ...,
    schema_format = c("ollama", "litellm"),
    json_name = "annotation"
) {
  
  # Match schema format
  schema_format <- match.arg(schema_format)
  
  # Record whether json_name was explicitly supplied
  json_name_supplied <- !missing(json_name)
  
  # Collect annotation fields
  fields <- list(...)
  
  # Validate annotation fields
  if (length(fields) == 0L) {
    stop(
      "At least one annotation category is required.",
      call. = FALSE
    )
  }
  
  field_names <- names(fields)
  
  if (is.null(field_names) ||
      anyNA(field_names) ||
      any(!nzchar(field_names)) ||
      anyDuplicated(field_names)) {
    stop(
      "All annotation categories must have unique, ",
      "non-empty names.",
      call. = FALSE
    )
  }
  
  # Convert a field into a JSON Schema property
  build_property <- function(field, name) {
    
    # Open-ended field
    if (inherits(field, "bac_field")) {
      
      if (!is.list(field) ||
          !identical(names(field), "type") ||
          !is.character(field$type) ||
          length(field$type) != 1L ||
          is.na(field$type) ||
          !field$type %in% c(
            "string", "number", "integer", "boolean"
          )) {
        stop(
          "Invalid open-field specification for `",
          name, "`.",
          call. = FALSE
        )
      }
      
      return(unclass(field))
    }
    
    # Closed category with predefined labels
    if (is.character(field) &&
        length(field) > 0L &&
        !anyNA(field) &&
        all(nzchar(field)) &&
        !anyDuplicated(field)) {
      
      return(list(
        type = "string",
        enum = as.list(field)
      ))
    }
    
    stop(
      "Invalid specification for category `", name,
      "`. Supply a non-empty character vector of unique ",
      "labels or a bac_open_*() field.",
      call. = FALSE
    )
  }
  
  # Build provider-independent JSON Schema
  properties <- Map(
    f = build_property,
    field = fields,
    name = field_names
  )
  
  names(properties) <- field_names
  
  schema <- list(
    type = "object",
    properties = properties,
    required = as.list(field_names),
    additionalProperties = FALSE
  )
  
  # Ollama format
  if (schema_format == "ollama") {
    
    if (json_name_supplied) {
      warning(
        "`json_name` is ignored when ",
        "`schema_format = 'ollama'`.",
        call. = FALSE
      )
    }
    
    return(schema)
  }
  
  # LiteLLM format
  if (!is.character(json_name) ||
      length(json_name) != 1L ||
      is.na(json_name) ||
      !grepl("^[A-Za-z0-9_-]{1,64}$", json_name)) {
    stop(
      "`json_name` must contain 1-64 characters ",
      "using letters, digits, underscores, or hyphens.",
      call. = FALSE
    )
  }
  
  list(
    type = "json_schema",
    json_schema = list(
      name = json_name,
      strict = TRUE,
      schema = schema
    )
  )
}
