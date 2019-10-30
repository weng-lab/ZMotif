library(rjson)
library(ggseqlogo)
library(ggplot2)
library(cowplot)

args = commandArgs(trailingOnly=TRUE)
input_json = args[1]
results_file = args[2]

get_information_content <- function(ppm) {
  pseudo_count = .001
  return(apply(ppm, 2, function(x) x[1]*log2((x[1] + pseudo_count)/.25) + 
                 x[2]*log2((x[2] + pseudo_count)/.25) + x[3]*log2((x[3] + pseudo_count)/.25) + 
                 x[4]*log2((x[4] + pseudo_count)/.25)))
}

trim_ppm <- function(ppm, min_info) {
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

get_gc_content <-  function(ppm) {
  w = ncol(ppm)
  return(sum(ppm[2:3,]) / w)
}


data <- fromJSON(file = input_json)
ppms = list()
plot_list = list()
tfs = list()
ids = list()
index = 1
for (entry in data) {
  if (entry[[2]] > 100){
    print(index)
    ppm <- t(matrix(as.numeric(unlist(entry[[3]])), ncol = 4, byrow = TRUE))
    row.names(ppm) <- c("A", "C", "G", "T")
    ppm <- trim_ppm(ppm = ppm, min_info = 0.1)
    #ppm <- get_rc(ppm = ppm)
    consensus_seq <- get_consensus_sequence(ppm)
    ppms[[index]] <- ppm
    tf = entry[[4]][[1]]
    tf = strsplit(tf, "_")[[1]][[1]]
    id = entry[[1]]
    n_sites = entry[[2]]
    p_value = entry[[5]][[1]]
    title = sprintf("%s: n = %d; p = %.2e\nid = %s; %s", tf, n_sites, p_value, id, consensus_seq)
    p <- ggseqlogo(ppm) + 
      ggtitle(title) + 
      scale_y_continuous(name = "Bits",
                         breaks = seq(0.0, 2.0, 1.0),
                         limits=c(0.0, 2.0)) + 
      theme(title =element_text(size=8),
            axis.text.x = element_text(angle = 90)) + 
      annotate('segment', x = 0.5, xend=0.5, y=0.0, yend=2.0, size=1.5) + 
      annotate('segment', x = 0.5, xend=ncol(ppm) + 0.5, y=0.0, yend=0.0, size=1.5)
    plot_list[[index]] <- p
    ids[[index]] = id
    tfs[[index]] = tf
    index = index + 1
  }
  
}
pdf(file = results_file, height = 11, width = 8.5)
cowplot::plot_grid(plotlist = plot_list, ncol = 3, nrow = 5) 
dev.off()

