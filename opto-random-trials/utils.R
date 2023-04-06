# Parsing BIDS ------------------------------------------------------------

parse_subject <- function(session_folder) {
  file <- list.files(path = session_folder, pattern = "^sub-.+-")
  assertthat::assert_that(
    assertthat::not_empty(file),
    msg = glue::glue("No subject files found in {session_folder}")
  )
  subject <- strsplit(file[1], split = "-|_")[[1]][2]
  return(subject)
}


# Plots -------------------------------------------------------------------


plot_hypno <- function(behavior, behavior_levels, trials) {
  trial_colors <- c("opto" = "#149DFF", "sham" = "gray")
  p <- ggplot() +
    add_trials(trials) +
    geom_line(data = behavior,
              aes(
                cam_timestamps / 3600,
                factor(behavior, levels = behavior_levels),
                group = 1
              ),
              size = 0.8) +
    scale_fill_manual(values = trial_colors) +
    scale_color_manual(values = colorspace::darken(trial_colors)) +
    labs(x = "Time (h)", y = "")
  return(p)
}

seconds_to_hours <- function(x) {
  x / 3600
}

add_trials <- function(trials) {
  trials <- trials %>%
    mutate(dplyr::across(.cols = c(onset, offset), .fns = seconds_to_hours))
  return(list(
    geom_vline(
      data = trials,
      mapping = aes(xintercept = onset, color = trial_type),
      show.legend = F
    ),
    geom_rect(
      data = trials,
      mapping = aes(
        xmin = onset,
        xmax = offset,
        ymin = -Inf,
        ymax = Inf,
        fill = trial_type
      ),
      alpha = 0.5
    )
  ))
}

lag_df_crop <- function(df, behavior_levels) {
  df %>%
    mutate(behavior = factor(behavior, levels = behavior_levels)) %>%
    group_by(trial_n) %>%
    mutate(lg = lag(behavior, default = "first frame"),
           # check if there's continuity
           flag = lg != behavior) %>%
    filter(flag) %>%
    mutate(time_end = lead(cam_timestamps, 1)) %>%
    # align all time axes
    mutate(
      cam_shifted = cam_timestamps - first(cam_timestamps),
      time_end_shifted = time_end - first(cam_timestamps),
      # the last one has to go all the way to the end of the trial
      time_end_shifted = if_else(is.na(time_end_shifted), 300, time_end_shifted)
    )
}


# This function will make light shading pattern for ggplot plots
#' @param params The params from `config.yaml` that will tell what the target dates are and what the light schedule is.
#' @param df A `data.frame` containing a `datetime` column that will provide the ranges of the experimental data

make_lights <- function(params, df) {
  # get the range of the experiment
  # this might fail if no FED has been tried before, shouldn't be a problem if we test all feds can deliver a pellet before putting them in the cage
  experiment_range <- range(df$datetime)
  
  params$fed_dates <- seq(lubridate::date(experiment_range[1]),
                          lubridate::date(experiment_range[2]),
                          "1 day")
  
  #make the dates
  # this vector will have on, off, on, off..
  # the first value might be before the experiment starts, the last value after experiment ends
  light_changes <-
    lapply(params$fed_dates, function(tt)
      paste(tt, params$lights)) %>%
    unlist()
  #make it datetime
  light_changes <- lubridate::as_datetime(light_changes)
  # subset
  light_changes <-
    light_changes[data.table::between(light_changes, experiment_range[1], experiment_range[2])]
  
  light_diff <- diff(hms::as_hms(params$lights))
  
  # we need even number light changes to make rectangles
  # light changes could be odd when starting and finishing animals at different light cycles
  if (length(light_changes) %% 2 == 1) {
    # It might happen that the first light change is lights-on because
    # the animals started during on lights off
    # check for that
    # if the first time is lights on, add the previous lights off
    if (hms::as_hms(params$lights[1]) == hms::as_hms(light_changes[1])) {
      # calculate the beginning of the lights-off
      # first lights off will be the previous day
      first_lights_off <- dplyr::first(light_changes) - light_diff
      # it was done like this previously "light_changes[1] - lubridate::hours(12)"
      # but it might not be a 12 hs cycle
      light_changes <-
        purrr::prepend(light_changes, first_lights_off)
      print(glue::glue("Adding {first_lights_off}"))
      # the xmin should be the even one in this case
      full_exp_shade <-
        annotate(
          "rect",
          xmin = light_changes[seq_along(light_changes) %% 2 == 0],
          xmax = light_changes[seq_along(light_changes) %% 2 > 0],
          ymin = 0,
          ymax = Inf,
          fill = "gray80",
          alpha = 0.5
        )
    } else {
      # if the first value is lights-off, add the last lights-on
      last_lights_on <- dplyr::last(light_changes) + light_diff
      light_changes <-
        append(light_changes, values = last_lights_on)
      full_exp_shade <-
        annotate(
          "rect",
          xmin = light_changes[seq_along(light_changes) %% 2 > 0],
          xmax = light_changes[seq_along(light_changes) %% 2 == 0],
          ymin = 0,
          ymax = Inf,
          fill = "gray80",
          alpha = 0.5
        )
    }
  } else {
    full_exp_shade <-
      annotate(
        "rect",
        # the xmin should be the odd one in this case
        xmin = light_changes[seq_along(light_changes) %% 2 > 0],
        xmax = light_changes[seq_along(light_changes) %% 2 == 0],
        ymin = 0,
        ymax = Inf,
        fill = "gray80",
        alpha = 0.5
      )
  }
  
  return(full_exp_shade)
}


add_epoc <- function(joint_pellet_data, experiment_epocs) {
  assertthat::assert_that(assertthat::has_name(joint_pellet_data, "tdt_datetime"))
  joint_pellet_data %>%
    mutate(
      epoc = case_when(
        data.table::between(tdt_datetime,
                            exp_start,
                            experiment_epocs["opto_start"]) ~ "pre",
        data.table::between(tdt_datetime,
                            experiment_epocs["opto_start"],
                            experiment_epocs["opto_stop"]) ~ "stim",
        data.table::between(tdt_datetime,
                            experiment_epocs["opto_stop"],
                            exp_stop) ~ "post",
        TRUE ~ "hab"
      ),
      epoc = factor(epoc,  levels = c("pre", "stim", "post"))
    )
}


# Behavior analysis -------------------------------------------------------

summarise_by_epoc <- function(df) {
  assertthat::assert_that(assertthat::has_name(df, "epoc"))
  df %>%
    dplyr::group_by(epoc, .drop = F) %>%
    dplyr::count()
}

bin_behavior <- function(df, time_sec, bin_sec) {
  max_t = max(dplyr::pull(df, {
    {
      time_sec
    }
  }))
  breaks <- seq(from = 0,
                to = max_t + bin_sec,
                by = bin_sec)
  # the function mfv returns the most frequent value
  # it there are ties, it returns the first of the n ties, not perfect but good enough
  df %>%
    dplyr::mutate(bin = cut({
      {
        time_sec
      }
    }, breaks = breaks)) %>%
    dplyr::group_by(bin) %>%
    dplyr::mutate(mfv1 = statip::mfv1(behavior)) %>%
    dplyr::ungroup()
}

# This function cleans the names after using a `cut` on a column
clean_cut_labels <- function(cut_labels) {
  mat <-
    stringr::str_split(
      string = stringr::str_sub(cut_labels, start = 2, end = -2),
      pattern = ",",
      simplify = TRUE,
      n = 2
    )
  mat <- apply(mat, 2, as.numeric)
  colnames(mat) <- c("low", "high")
  return(mat)
}


filter_between <- function(df, col, lower, upper) {
  assertthat::assert_that(assertthat::is.number(lower))
  assertthat::assert_that(assertthat::is.number(upper))
  assertthat::assert_that(lower < upper)
  df %>%
    dplyr::filter(data.table::between({{col}}, lower, upper))
}
