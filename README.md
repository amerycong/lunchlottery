# lunchlottery

This is my brute force impementation of a variation of the Kirkman Schoolgirl/Social Golfer problem. My approach was to generate a large number of permutations of groups and testing each one for uniqueness and lowest cost. 

Can be run by simply calling lottery.py from within ipython. Parameters (hardcoded) listed below. 

-normal_offset and same_group_offset are the values at which cost matrices are initialized at for a given pair of people, and the value by which the existing value is increased for each saved matchup. 

-save_dir is the directory in which everything is run from and stored in 

-num_per_group is the number of people in an ideal group, this number is increased by 1 automatically for big groups to account for remainders 

-min_unique is the total number of lotteries a person can go through guaranteed without a rematch, a value of x means that x-1 previous matchups will be stored 

RUNTIME 

-make sure the save_dir is pointed at the correct path and that responses.csv of the most recent lottery exists in that path, as well as lottery_info.pkl (if it exists) 
-just run lottery.py 
-you will have the option to check the generated group before continuing, and can continue generating a possibly better group or can publish (save all important variables and send emails) 

NOTES 

-if you are running overtime too much, try reducing min_unique

Here's a visualization of an example cost matrix - the main optimization variable in the algorithm. Our method aims to choose groups to minimize the total costs (darker squares = higher costs) according to our network. As expected, the self-cost is infinite since you can't group with yourself, and the values are symmetric about the diagonal.

![costs](visualmatrix.png?raw=true "Cost Matrix Visualization")
