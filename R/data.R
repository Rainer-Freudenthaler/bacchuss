#' Example fewshot examples dataset
#'
#' A dataset containing few-shot examples for testing functions.
#'
#' @format A data frame with 50 rows and 3 columns:
#' \describe{
#'   \item{input}{few shot example test}
#'   \item{explanation}{explanation for the true label}
#'   \item{label}{true label}
#' }
"example_fewshot"

#' Annotation instructions for explanation mode
#'
#' Example instructions used for annotation with bacchuss.
#'
#' @format A character vector.
"example_instructions_expl"

#' Annotation instructions without explanation
#'
#' Example instructions used for annotation with bacchuss.
#'
#' @format A character vector.
"example_instructions"

#' Annotation reminder with explanation
#'
#' Example reminder used for annotation with bacchuss.
#'
#' @format A character vector.
"example_reminder_expl"

#' Annotation reminder without explanation
#'
#' Example reminder used for annotation with bacchuss.
#'
#' @format A character vector.
"example_reminder"

#' Example fewshot dataset
#'
#' A dataset containing human annotated data to test annotation functions.
#'
#' @format A data frame with 809 rows and 5 columns:
#' \describe{
#'   \item{paragraphs}{text for annotation}
#'   \item{group}{the group mentioned in the text}
#'   \item{Emotion}{true label from human annotation}
#'   \item{length}{word count of text}
#'   \item{estimated context length}{estimated context length for annotation}r
#' }
"example_set"