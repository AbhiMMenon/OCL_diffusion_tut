
#set size square
set multiplot layout 1,2
set title "Initial"
set xlabel r"x"
set ylabel r"y"
plot "init1.dat" binary with image 
set title "Final"
plot "out.dat" binary  with image 

