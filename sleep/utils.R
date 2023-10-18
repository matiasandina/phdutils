# This function helps import sleep labels saved into .mat by Accusleep
# the labels are buried inside the $labels as an nx1 matrix, we want to output a vector
import_mat_labels <- function(filepath, convert = TRUE){
  labels <- as.vector(R.matlab::readMat(filepath)$labels)
  if (convert){
    labels <- convert_accusleep_labels(labels)
  }
  return(labels)
}

convert_accusleep_labels <- function(col){
  assertthat::assert_that(is.numeric(col),
                          msg = glue::glue("`col` must be numeric, received `{class(col)}`"))
  return(
    dplyr::case_when(col == 1 ~ "REM",
              col == 2 ~ "Wake",
              col == 3 ~ "NREM")
    )
  }



convert_yasa_int <- function(col){
  # We are doing this conversion
  # '[0, 1, 2, 3, 4] ==> [‘W’, ‘N1’, ‘N2’, ‘N3’, ‘R’]'
  assertthat::assert_that(is.integer(col),
                          msg = glue::glue("`col` must be integer, received `{class(col)}`"))
  return(
    dplyr::case_when(col == 0L ~ "Wake",
                     # All 'N' go to NREM
                     col == 1L ~ "NREM",
                     col == 2L ~ "NREM",
                     col == 3L ~ "NREM",
                     col == 4L ~ "REM")
  )
}


convert_yasa_strings <- function(col){
  assertthat::assert_that(is.character(col),
                          msg = glue::glue("`col` must be character, received `{class(col)}`"))
  return(
    dplyr::case_when(col == 'R' ~ "REM",
                     col == 'W' ~ "Wake",
                     str_detect(col, "N") ~ "NREM")
  )
}


make_time_column <- function(sf, ...){
  return(seq(from = 0, by = 1/sf, ... ))
}

seconds_to_hours <- function(x){x/3600}

# This comes handy for the arrows in facets and whatnot
paste_behavior <- function(pre, post){
  paste(pre, "→", post)
}


# reading sleep functions -------------------------------------------------
read_sleep_from_mat <- function(filepath, params, scoring_period = 2, convert_accusleep=TRUE){
  eeg_t0_sec <- pluck(params, 'eeg_t0_sec', 'value')
  max_t <- pluck(params, 'photo_max_t', 'value')
  sleep_behavior <- tibble(
    sleep = import_mat_labels(filepath, convert=convert_accusleep),
    behavior = ifelse(sleep == "Wake", sleep, "Sleep")) %>% 
    mutate(time_sec = make_time_column(sf = 1/scoring_period,
                                       length.out=n()), 
           aligned_time_sec = time_sec - eeg_t0_sec)  %>% 
    filter(data.table::between(aligned_time_sec, 0, max_t))
  
  # Microarousals
  sleep_behavior <- sleep_behavior %>% 
    mutate(run_id = vctrs::vec_identify_runs(sleep)) %>% 
    mutate(.by="run_id", 
           duration = last(aligned_time_sec) - first(aligned_time_sec) + scoring_period, 
           sleep2 = if_else(sleep == "Wake" & duration < 16, "MA", sleep)) 
  return(sleep_behavior)
}


read_sleep_consensus <- function(filepath, params, scoring_period = 2){
  eeg_t0_sec <- pluck(params, 'eeg_t0_sec', 'value')
  max_t <- pluck(params, 'photo_max_t', 'value')
  sleep_behavior <- readr::read_csv(filepath, show_col_types = FALSE) %>% 
    mutate_all(.funs = convert_yasa_strings) %>% 
    mutate(time_sec = make_time_column(sf = 1/scoring_period,
                                       length.out=n()), 
           aligned_time_sec = time_sec - eeg_t0_sec)  %>% 
    filter(data.table::between(aligned_time_sec, 0, max_t)) %>% 
    # rename to get an extra column
    # we can repurpose if we wanted to switch to mfv1
    mutate(sleep = consensus)
  
  # Microarousals
  sleep_behavior <- sleep_behavior %>% 
    mutate(run_id = vctrs::vec_identify_runs(sleep)) %>% 
    mutate(.by="run_id", 
           duration = last(aligned_time_sec) - first(aligned_time_sec) + scoring_period, 
           sleep2 = if_else(sleep == "Wake" & duration < 16, "MA", sleep)) 
  return(sleep_behavior)
}

plot_eeg_array <- function(ap, ml){
  df <- dplyr::arrange(tidyr::expand_grid(ap, ml), desc(ap))
  df$label <- paste("channel",1:nrow(df)) 
  ggplot(df, aes(x=ml, y=ap)) +
    geom_hline(yintercept = 0, lty = 2) +
    geom_vline(xintercept = 0, lty = 2) +
    geom_point(pch=19, size = 5) +
    geom_text(aes(label=label), size = 5, nudge_y = 0.6) +
    scale_x_continuous(limits = c(-3.5, 3.5), )+
    scale_y_continuous(limits = c(-5, 5))
}

smooth_mode <- function(x, width=3){
  zoo::rollapply(data = x, 
                 align = "center",
                 width = width,
                 FUN = collapse::fmode, 
                 # partial = TRUE keeps the ends to be same length
                 partial=TRUE)
}


# Plotting ----------------------------------------------------------------
# This is not a multi-taper spectrogram but it's OK
spectro <- function(data, sf, nfft=1024, window=256, overlap=128, t0=0, plot_spec = T, normalize = F, return_data = F, ...){
  
  # create spectrogram
  spec = signal::specgram(x = data,
                          n = nfft,
                          Fs = sf,
                          window = window,
                          overlap = overlap
  )
  
  # discard phase info
  S = as.data.frame(abs(spec$S))
  # normalize
  if(normalize){
    S = S/max(S)  
  }
  
  # name S
  names(S) <- spec$t
  # add freq
  S$f = spec$f
  
  # pivot longer
  S <- S %>% pivot_longer(-f, names_to = "time", values_to = "power") %>% 
    mutate(time = as.numeric(time))
  
  # config time axis
  if (is.numeric(t0)) {
    S <- mutate(S, time = time + t0)
  } else if (is.POSIXct(t0)){
    S <- mutate(S, time = as.POSIXct(time, origin = t0))
  } else{
    stop("t0 must be either `numeric` or `POSIXct`")
  }
  
  out_plot <- 
    ggplot(S, aes(time, f, fill = power)) + 
    geom_tile(show.legend = F) +
    scale_fill_viridis_c(...) +
    labs(y = "Freq (Hz)") 
  return(out_plot)
}

plot_trace <- function(df, x, y, trange, ...){
  p1 <- ggplot(df %>% 
                 filter(data.table::between({{x}},
                                            trange[1],
                                            trange[2])
                 ),
               aes(x={{x}}, y={{y}})) + 
    geom_line(...) + 
    xlab("")
  return(p1)
}

get_ethogram <- function(data, x, behaviour, sampling_period = NULL){
  if (is.null(sampling_period)){
    cli::cli_alert_warning("`sampling_period` not provided.")
    sampling_period <- min(diff(dplyr::pull(data, {{x}})))
    cli::cli_inform("Sampling period estimated to {sampling_period} using min difference between observations")
  }
  
  if(dplyr::is_grouped_df(data)){
    cli::cli_alert_info("Data was grouped by {dplyr:::group_vars(data)}")
    data <- dplyr::select(data, dplyr::group_cols(), x = {{x}}, behaviour = {{behaviour}})
  } else {
    data <- dplyr::select(data, x = {{x}}, behaviour = {{behaviour}}) 
  }
  
  etho <- data %>% 
    dplyr::mutate(run_id = vctrs::vec_identify_runs(behaviour)) %>% 
    # add to whatever previous layer was there
    group_by(run_id, .add=TRUE) %>% 
    dplyr::summarise(behaviour = base::unique(behaviour), 
                     xend = dplyr::last(x) + sampling_period, 
                     x = dplyr::first(x), 
                     duration = xend - x, 
                     .groups = "keep") %>% 
    dplyr::select(dplyr::group_cols(), x, xend, behaviour, duration)
  
  return(etho)
}

# This function is intended to grab outside data and filter passing a nested .x with x$x and x$xend
filter_tranges <- function(data, .x, right_end = NULL){
  if (is.null(right_end)){
    # use all signal
    right_end <- .x$xend + t_delta
  } else {
    # use up to right_end in seconds
    right_end <- min(.x$x + right_end, .x$xend + t_delta)
  }

   data %>%  
    filter(data.table::between(aligned_time_sec, 
                               lower = max(0, .x$x - t_delta), 
                               upper = min(max_t, right_end))) %>%
    mutate(rel_time = aligned_time_sec - dplyr::first(aligned_time_sec) - t_delta)

}

# This function is useful to bind a tibble nested by run_id and get the photometry and behavior traces aligned in time
# .x is in the form of trange with x and xend for each run_id. 
# run this function inside a mutate(snips = map(tranges, ...))
filter_between_join_behavior <- function(data, sleep_data, sleep_col, .x, right_end = NULL) {
  
    filter_tranges(data, right_end = right_end, .x = .x) %>% 
    left_join(select(sleep_data, aligned_time_sec, {{sleep_col}}), 
              # we need a rolling join here because the two time columns will not be identical (numerical precision)
              # photometry time >= behavior time is key to avoid off by-one errors
              by = join_by(closest(aligned_time_sec >= aligned_time_sec)))
}


# This function checks a particular run by filtering
check_run_time <- function(data, id){
  data %>%
    ungroup() %>% 
    filter(run_id == id) %>% 
    unnest(tranges) %>% 
    mutate(x = hms::as_hms(x + eeg_t0_sec)) %>% 
    pull(x)
}


# 
library(pracma)
library(dplyr)

compute_hilbert <- function(electrode, sampling_frequency) {
  bands <- list(
    "delta" = c(0.5, 4),
    "theta" = c(4, 8),
    "sigma" = c(8, 15)
  )
  
  envelopes <- list()
  for (band in names(bands)) {
    low <- bands[[band]][1]
    high <- bands[[band]][2]
    
    # Apply bandpass filter
    wpass <- c(low, high) / (sampling_frequency / 2)
    sos <- gsignal::butter(10, wpass, type='pass', output='Sos')
    filtered <- gsignal::sosfilt(sos$sos, electrode)
    print(filtered[1:5])
    
    # Apply Hilbert transform to get the envelope (i.e., the amplitude) of the signal
    analytic_signal <- gsignal::hilbert(filtered)
    amplitude_envelope <- Mod(analytic_signal)
    
    # Store the envelope in the DataFrame
    envelopes[[band]] <- amplitude_envelope
  }
  
  envelopes <- dplyr::bind_cols(envelopes)
  return(envelopes)
}
