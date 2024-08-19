parse_pills <- function(header_line) {
  Encoding(header_line) <- "latin1"
  # Now that the encoding is set correctly, we can replace the "Pill n°" with "Pill"
  clean_header <- stringr::str_replace_all(header_line, "Pill n°", "Pill")

  # Extract matches for pill identifiers and their associated IDs
  pill_pattern <- "Pill([0-9]{1}),([0-9A-Z.]+)"
  matches <- stringr::str_extract_all(clean_header, pill_pattern)[[1]]
  
  # If no matches are found, return empty lists
  if (length(matches) == 0) {
    stop("No Pills found in dataset. Check pattern and encoding to be latin1")
  }
  
  # let's return a tibble of pills and unique ids
  output <- matches %>% 
    tibble::enframe(name = NULL) %>% 
    tidyr::separate(value, into = c("pill_n", "pill_id"), sep = ",")
  return(output)
}

process_header_metadata <- function(raw_data) {
  # extract the headers
  header_metadata_lines <- which(grepl("UTC|Pill", raw_data))
  # Split the header line into components
  # first line header_metadata_lines[1] has the timezone
  timezone_line <- raw_data[header_metadata_lines[1]]
  cleaned_header <- str_split(timezone_line, ",") %>%
    unlist() %>% 
    str_replace_all(pattern  = " ", replacement = "")
    # Remove any empty strings that may be due to trailing commas
    # and remove spaces

  # Create a named vector or list with the timezone information
  metadata <- setNames(list(cleaned_header[2]), cleaned_header[1])
  
  # now let's do the second component (the pill number)
  
  header_line <- raw_data[header_metadata_lines[2]]
  
  pill_metadata <- parse_pills(header_line)
  metadata$pill_metadata <- pill_metadata
  return(metadata)
}

# Function to check if a line is metadata
is_tail_metadata <- function(line) {
# this is dependent on the date format selected so expect errors
# we are going to try to parse dd/mm/yyyy,hh:mm:ss,(SAMPLING PERIOD CHANGED or -)
grepl("^\\d{2}/\\d{2}/\\d{4},\\d{2}:\\d{2}:\\d{2},(SAMPLING PERIOD CHANGED| - )",
      line)
}

parse_tail_metadata <- function(raw_data){
  # TODO: this depends heavily on datetime format selected
  tail_metadata_lines <- which(map_lgl(raw_data, is_tail_metadata))
  tail_metadata <- raw_data[tail_metadata_lines]
  tail_metadata %>% 
    enframe(name=NULL) %>% 
    # remove trailing comma at the end
    mutate(value = str_replace(value, ",$", ""),) %>% 
    separate(value, into=c('date', 'time', 'event'), ',') %>% 
    # TODO: HERE FOR EXAMPLE HEAVILY DEPENDENT ON FORMAT
    mutate(datetime = dmy_hms(paste(date, time)),
           # keep the idx
           raw_data_idx = tail_metadata_lines
           )
}


parse_data_from_raw <- function(raw_data, metadata){
  # data begins at the "Sample number" line
  sample_number_line <- which(grepl("Sample number", raw_data))
  # data ends with two blank lines (we )
  data_end_line <- min(metadata$tail_metadata$raw_data_idx) - 3
  data_to_clean <- raw_data[sample_number_line:data_end_line]
  return(data_to_clean)
}


parse_pill_data <- function(pill_data, pill_metadata) {
  # List to store data frames for each pill
  pill_dfs <- list()
  # we use read_csv to convert to a tibble
  # we expect the line by line character vector that can be \n pasted like this
  pill_data <-  read_csv(file = paste(pill_data, collapse = "\n"),
                         show_col_types = FALSE,
                         col_names = FALSE) %>% 
    # let's also remove the first row since it's the header
    slice(-1)

  # Number of columns per pill in the dataset
  cols_per_pill <- 5
  
  # Iterate over the number of pills based on metadata
  for (i in seq_len(nrow(pill_metadata))) {
    # Calculate start and end indices for each dataset
    start_idx <- 1 + (i - 1) * (cols_per_pill + 2)
    end_idx <- start_idx + cols_per_pill - 1
    
    # Extract the data for the current pill
    pill_df <- pill_data %>%
      # Select columns for the current pill; handle empty columns by filtering out
      select(all_of(start_idx:end_idx)) %>%
      # Set the column names
      set_names(c("sample_number", "date", "time", "temperature", "status")) %>%
      # Remove rows that are completely NA (if any due to misalignment)
      filter(!if_all(everything(), is.na)) %>%
      # Add pill identifier
      mutate(pill_n = pill_metadata$pill_n[i])
    
    # Append the dataframe to the list
    pill_dfs[[i]] <- pill_df
  }

  # Bind all data frames together into a single data frame
  final_df <- bind_rows(pill_dfs)

  return(final_df)
}


# This the actual functin that you would call
import_raw_anipill_data<-function(filepath){
  raw_data <- read_lines(filepath)
  metadata <- process_header_metadata(raw_data)
  metadata$tail_metadata <- parse_tail_metadata(raw_data)
  data_to_clean <- parse_data_from_raw(raw_data, metadata) 
  data_out <- parse_pill_data(pill_data = data_to_clean,
                              pill_metadata = metadata$pill_metadata)
  # clean data out
  data_out <- mutate(data_out,
                     # TODO: format dependent
                     datetime = dmy_hms(paste(date,time)),
                     temperature = as.numeric(temperature),
                     sample_number = as.numeric(sample_number))
  return(list(metadata=metadata, data = data_out))
}

# This can also be a public API function 
# since user might want to grab markers from metadata
grab_markers <- function(metadata){
  metadata$tail_metadata %>% filter(event == ' - ') %>% pull(datetime)
}
