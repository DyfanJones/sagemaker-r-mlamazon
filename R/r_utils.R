`%||%` <- function(x, y) if (is.null(x)) return(y) else return(x)

# Correctly mimic python append method for list
# Full credit to package rlist: https://github.com/renkun-ken/rlist/blob/2692e064fc7b6cc7fe7079b3762df37bc25b3dbd/R/list.insert.R#L26-L44
list.append <- function (.data, ...) {
  if (is.list(.data)) c(.data, list(...)) else c(.data, ..., recursive = FALSE)
}

is_subset = function(A, B){
  all(unique(A) %in% unique(B))
}

fs_path = function(...){
  return(as.character(fs::path(...)))
}

join_path = function(...){
  args = list(...)
  paste(trimws(args, whitespace="/"), collapse="/")
}
