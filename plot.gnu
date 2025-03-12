set multiplot layout 1,2
set title "Initial"
set xlabel r"x"
set ylabel r"y"
set xrange [0:1]
set yrange [0:1]
plot "data/init1.dat" binary with image notitle
set title "Final"
plot "data/out.dat" binary  with image notitle

