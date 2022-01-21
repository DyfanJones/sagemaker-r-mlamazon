`%||%` <- function(x, y) if (is.null(x)) return(y) else return(x)

list.append <- function (.data, ...) {
  if (is.list(.data)) c(.data, list(...)) else c(.data, ..., recursive = FALSE)
}
