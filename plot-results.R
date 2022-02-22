library(rjson)
library(ggseqlogo)
library(ggplot2)
library(cowplot)

get_information_content <- function(ppm) {
  pseudo_count = .001
  return(apply(ppm, 2, function(x) x[1]*log2((x[1] + pseudo_count)/.25) + 
                 x[2]*log2((x[2] + pseudo_count)/.25) + x[3]*log2((x[3] + pseudo_count)/.25) + 
                 x[4]*log2((x[4] + pseudo_count)/.25)))
}

trim_ppm <- function(ppm, min_info) {
  tryCatch(
    {
      start_index = 1
      width = ncol(ppm)
      stop_index = width
      info <- get_information_content(ppm = ppm)
      #print(info)
      for (i in 1:width) {
        if (info[i] < min_info) {
          start_index = start_index + 1
        } else {
          break
        }
      }
      
      for (i in width:1) { 
        if (info[i] < min_info) {
          stop_index = stop_index - 1
        } else {
          break
        }
      }
      return(ppm[,start_index:stop_index])
    },
    error=function(e) {
      message(cond)
      # Choose a return value in case of error
      return(ppm)
    },
    warning=function(w) {
      message(cond)
      # Choose a return value in case of warning
      return(ppm)
    },
    finally={
      # NOTE:
      # Here goes everything that should be executed at the end,
      # regardless of success or error.
      # If you want more than one expression to be executed, then you 
      # need to wrap them in curly brackets ({...}); otherwise you could
      # just have written 'finally=<expression>' 
      #message("Some other message at the end")
    }
  )
}

get_consensus_sequence <- function(ppm) {
  seq = ""
  for (nuc in rownames(ppm)[apply(ppm, 2, which.max)]) {
    seq = paste(seq, nuc, sep = "")
  }
  return(seq)
}

get_rc <- function(ppm) {
  w = ncol(ppm)
  rc <- ppm[4:1,w:1]
  row.names(rc) <- c("A", "C", "G", "T")
  return(rc)
}

get_ga_content <-  function(ppm) {
  w = ncol(ppm)
  return(sum(ppm[c(1,3),]) / w)
}
  
args = commandArgs(trailingOnly=TRUE)
data <- fromJSON(file = args[[1]])

ppms = list()
plot_list = list()
tfs = list()
ids = list()
index = 1
for (entry in data) {
  if (entry$n_sites > 10){
    ppm <- t(matrix(as.numeric(unlist(entry$ppm)), ncol = 4, byrow = TRUE))
    row.names(ppm) <- c("A", "C", "G", "T")
    ppm <- trim_ppm(ppm = ppm, min_info = 0.1)
    ga = get_ga_content(ppm)
    print(ga)
    if (ga < 0.5) {
      ppm <- get_rc(ppm)
    }
    
    consensus_seq <- get_consensus_sequence(ppm)
    ppms[[index]] <- ppm
    match = entry$matches[[1]]
    id = entry$id
    n_sites = entry$n_sites
    p_value = entry$p_values[[1]]
    auroc = entry$auroc
    title = sprintf("%s\nn = %d; id = %s", match, n_sites, id)
    # title = sprintf("%s: n = %d; p = %.2e\nid = %s; %s", tf, n_sites, p_value, id, consensus_seq)
    # title = sprintf("%s: n = %d; p = %.2e\n id = %s; auROC = %.3f", tf, n_sites, p_value, id, auroc)
    p <- ggseqlogo(ppm) + 
      ggtitle(title) +
      scale_y_continuous(name = "Bits",
                         breaks = seq(0.0, 2.0, 1.0),
                         limits=c(0.0, 2.0)) + 
      theme(axis.text.x = element_blank(),
            axis.title.y = element_blank(),
            axis.line.y = element_line(),
            axis.ticks.y = element_line(),
            title = element_text(size=4))
    plot_list[[index]] <- p
    index = index + 1
  }
  
}
pdf(args[[2]], width = 8, height = 4)
cowplot::plot_grid(plotlist = plot_list, ncol = 4, nrow = 4) 
dev.off()