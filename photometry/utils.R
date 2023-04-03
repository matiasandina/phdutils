# airPLS functions just in
devtools::source_url("https://raw.githubusercontent.com/zmzhang/airPLS_R/master/R/airPLS.R")

clean_cut_labels <- function(cut_labels){
  mat <- stringr::str_split(string = stringr::str_sub(cut_labels, start = 2, end = -2), 
                            pattern = ",",
                            simplify = T,
                            n = 2)
  mat <- apply(mat, 2, as.numeric)
  colnames(mat) <- c("low", "high")
  return(mat)
}

# The idea is that this function helps us match which photometry session we should use given the session folder contains the date
parse_photometry_session <- function(session_folder, photo_folders){
  # photo_folders is a vector which will have MLA[0-9]+-YYMMDD-HHMMSS/
  # session folder is a character that comes from params$session_folder with YYYY-MM-DD
  parsed_dates <- sapply(photo_folders,
                         function(xx) as.character(as.Date(stringr::str_split(xx, "-")[[1]][2],"%y%m%d"))
  )
  matched_folders <- str_detect(session_folder, pattern=parsed_dates)
  if (any(matched_folders)) {
    return(file.path(session_folder, photo_folders[which(matched_folders)]))
  } else {
    cli::cli_alert_danger("No patterns found using `session_folder` {session_folder} and `photo_folders` {photo_folders}")
  }
}
