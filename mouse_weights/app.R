library(shiny)
library(ggplot2)
library(gridlayout)
library(bslib)
library(shinyFiles)
library(shinyWidgets)
library(tibble)
library(DT)

plot_weight_data <- function(mouse_data){
  ggplot(
    mouse_data,
    aes(
      x = datetime,
      y = weight,
      group = id
    )
  ) +
    geom_line(alpha = 0.5) +
    ggtitle("Mouse weight over time") +
    theme_linedraw()
}


ui <- navbarPage(
  shinyjs::useShinyjs(),
  title = "🐁Mouse Weights",
  selected = "Line Plots",
  collapsible = TRUE,
  theme = bslib::bs_theme(),
  tabPanel(
    title = "Line Plots",
    grid_container(
      layout = c(
        "ui_in linePlots"
      ),
      gap_size = "16px",
      col_sizes = c(
        "350px",
        "1fr"
      ),
      row_sizes = c(
        "1fr"
      ),
      grid_card(
        area = "ui_in",
        card_header("Text here"),
        card_body_fill(
          shinyFilesButton('loadFileButton', 'Load Previous Data', title='Please choose a file', multiple=F, buttonType='default'),
          #fileInput("loadFile", "Load Previous Data", accept = c('.csv')),
          shinyDirButton('folder', 'Select/Create New Data directory', 'Upload'),
          verbatimTextOutput("folder_text"),
          verbatimTextOutput('mouse_id', placeholder = T),
          numericInput(
            inputId = "weightInput",
            label = "Enter Weight (gr)",
            value = 0,
            width = "100%"
          ),
          "Enter DateTime or click to select now:",
          actionButton(
            inputId = "nowButton",
            label = "Now",
            width = "20%"
          ),
          airDatepickerInput(
            inputId = "dateTimeInput",
            label = "",
            value = Sys.time(),
            timepicker = TRUE,
            todayButton = FALSE,
            update_on = "close",
            timepickerOpts = timepickerOptions(dateTimeSeparator = " ")
          ),
          textInput("notes", label = "Notes:", ""),
          actionButton("submitButton", "Submit Entry", width = "50%", 
                       icon = icon("ok", lib = "glyphicon", class="success")),
          actionButton("saveButton", "Save Data", width = "50%"),
          actionButton("clearData", "Clear Data", width = "50%", 
                       icon = icon('trash', lib = "glyphicon", class = 'error'))
        )
      ),
      grid_card_plot(area = "linePlots", width = "500px") # Reduced plot width
    )
  ),
  tabPanel("Data Table",
           DTOutput("dataTable"))
)


server <- function(input, output, session) {
  shinyjs::disable("submitButton")
  # Reactive value to keep track of whether the folder has been selected
  folderSelected <- reactiveVal(FALSE)
  loadButtonClicked <- reactiveVal(FALSE)
  selected_folder <- reactiveVal(NULL)
  file_path <- reactiveVal(NULL)
  animal_id <- reactiveVal(NULL)
  
  initial_tibble <- function() {
    tibble(
      id = character(),
      datetime = as.POSIXct(character()),
      weight = numeric(),
      note = character()
    )
  }
  
  # define the first time empty anyways for other initialization issues
  mouse_data <- reactiveVal(initial_tibble())

  observe({
    shinyDirChoose(input, 'folder', roots = c(home = normalizePath("~")))
    # If a folder is selected, update the reactive value
    if(inherits(input$folder, 'list')) {
      folder_path <- parseDirPath(c(home = normalizePath("~")), input$folder)
      selected_folder(folder_path)
      animal_id_value <- basename(folder_path)
      animal_id(animal_id_value)
      if (isFALSE(folderSelected())){
        folderSelected(TRUE)
        # define the data upon creation
        mouse_data(initial_tibble())
        print("Reseting Mouse Data")
        print(mouse_data())
      }
    }
  })
  
  observe({
    if(folderSelected()) {
      shinyjs::enable("submitButton")
    } else {
      shinyjs::disable("submitButton")
    }
  })
  
  observeEvent(animal_id(),{
    output$mouse_id <- renderText({
      paste("Animal ID is:", animal_id())
    })
  })

  observe({
    selected_folder_value <- selected_folder()
    if (!is.null(selected_folder_value)) {
      output$folder_text <- renderText({
        selected_folder_value
      })
    }
  })
  
  #observeEvent(input$loadFile, {
  #  file_path_value <- input$loadFile$datapath
  #  full_path <- file_path_value[1]
  #  folder_path <- dirname(full_path)
  #  animal_id_value <- basename(folder_path)
    
    # Assigning values to reactive variables
  #  file_path(file_path_value)
  #  selected_folder(folder_path)
  #  animal_id(animal_id_value)
    
  #  loaded_data <- readr::read_csv(full_path)
  #  mouse_data(loaded_data) # Update reactive value
    # Enable the submit button once the file is loaded
  #  shinyjs::enable("submitButton")
  #})
  
  observeEvent(input$loadFileButton, {
    shinyFileChoose(input, 'loadFileButton', roots = c(home = normalizePath("~")), filetypes = c('csv'))
  })
  
  observe({
    if (inherits(input$loadFileButton, "list")) {
      file_path_value <- parseFilePaths(c(home = normalizePath("~")), input$loadFileButton)
      full_path <- file_path_value$datapath[1]
      folder_path <- dirname(full_path)
      animal_id_value <- basename(folder_path)
      
      # Assigning values to reactive variables
      file_path(file_path_value)
      selected_folder(folder_path)
      animal_id(animal_id_value)
      
      loaded_data <- readr::read_csv(full_path)
      mouse_data(loaded_data) # Update reactive value
      # Enable the submit button once the file is loaded
      shinyjs::enable("submitButton")
    }
  })

  # Clear data
  observeEvent(input$clearData, {
    mouse_data(initial_tibble())
    shinyalert::shinyalert("Success", "Weight Data was Cleared!", type = "success")
  }
  )
  # Save data
  observeEvent(input$saveButton, {
    folder_path <- selected_folder() 
    animal_id <- basename(folder_path)
    save_name <- glue::glue("sub-{animal_id}_weights.csv")
    save_path <- file.path(folder_path, save_name)
    readr::write_csv(mouse_data(), save_path)
    shinyalert::shinyalert("Success", paste("Data saved to", save_path), type = "success")
  })
  
  observeEvent(input$nowButton, {
    updateAirDateInput(session, "dateTimeInput", value = Sys.time())
  })
  
  observeEvent(input$submitButton, {
    # Retrieve folder_path and animal_id from reactive values
    folder_path <- selected_folder()
    animal_id <- animal_id()
    # This is already POSIXct in UTC
    datetime <- input$dateTimeInput
    # Check for duplicate datetime for this id
    if (any(mouse_data()$id == animal_id & mouse_data()$datetime == datetime)) {
      shinyalert::shinyalert("Error", "This datetime is already recorded for this animal. Please choose another datetime.", type = "error")
      return()
    }
    
    if (input$weightInput == 0){
      shinyalert::shinyalert("Error", "Weight must be more than zero grams")
      return()
    }
    
    new_entry <- tibble(
      id = animal_id,
      datetime = datetime,
      weight = input$weightInput,
      note = input$notes
    )
    # Append new entry to mouse_data
    updated_data <- dplyr::bind_rows(mouse_data(), new_entry)
    # arrange by datetime
    updated_data <- dplyr::arrange(updated_data, datetime)
    mouse_data(updated_data) # Update reactive value
    output$dataTable <- DT::renderDT({
      pre <- mouse_data() %>% dplyr::select(id, datetime, everything())
      DT::datatable(pre) %>%
        # format utc_datetime in locale string instead of Z time ("toISOString")
        DT::formatDate(2, "toLocaleString")
    })
    if (nrow(mouse_data()) > 2) {
      p <- plot_weight_data(mouse_data())
      output$linePlots <- renderPlot({p})
    }
    # Clean the weight
    updateNumericInput(session, "weightInput", value = 0)
  })
  
  
  output$linePlots <- renderPlot({
    if (nrow(mouse_data()) < 2) {
      return("Less than two datapoints; graph will not be plotted.")
    } else {
      plot_weight_data(mouse_data())
    }

  })
  

}

shinyApp(ui, server)
