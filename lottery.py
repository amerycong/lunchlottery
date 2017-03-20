import csv
import numpy as np
import random
import math
import pickle
import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity

#USEFUL/RELEVANT LINKS
#http://mathworld.wolfram.com/KirkmansSchoolgirlProblem.html
#https://en.wikipedia.org/wiki/Kirkman%27s_schoolgirl_problem
#http://demonstrations.wolfram.com/SocialGolferProblem/
#http://www.metalevel.at/mst.pdf
#http://publications.cse.iitm.ac.in/775/1/WG98.pdf

'''
This is my brute force impementation of a variation of the Kirkman Schoolgirl/Social Golfer problem.
My approach was to generate a large number of permutations of groups and testing each one for uniqueness
and lowest cost. 
Can be run by simply calling lottery.py from within ipython. Parameters (hardcoded) listed below.
-normal_offset and same_group_offset are the values at which cost matrices are initialized at for a
given pair of people, and the value by which the existing value is increased for each saved matchup.
-save_dir is the directory in which everything is run from and stored in
-num_per_group is the number of people in an ideal group, this number is increased by 1 automatically
for big groups to account for remainders
-min_unique is the total number of lotteries a person can go through guaranteed without a rematch,
a value of x means that x-1 previous matchups will be stored

RUNTIME
-make sure the save_dir is pointed at the correct path and that responses.csv of the most recent lottery
exists in that path, as well as lottery_info.pkl (if it exists)
-just run lottery.py
-you will have the option to check the generated group before continuing, and can continue generating
a possibly better group or can publish (save all important variables and send emails)

NOTES
-if you are running overtime too much, try reducing min_unique
'''

########################################
############## PARAMETERS ##############
########################################

global normal_offset; normal_offset = 1
global same_group_offset; same_group_offset = 10
global save_dir; save_dir = '/home/amery/Documents'
num_per_group = 3 
min_unique = 3

########################################
########################################
########################################

def parse_responses(response_path):
	usernames=[]
	groups=[]
	names=[]

	#read csv
	with open(response_path,'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			usernames.append(row[1])
			groups.append(row[2])
			names.append(row[3])

	#format data
	usernames = usernames[1:]
	usernames = [name.split('@')[0] for name in usernames]
	groups = groups[1:]
	names = names[1:]

	#sort into groups (dicionary)
	groupnames = list(set(groups))
	errbody = {}
	for i in range(0,len(groupnames)):
		errbody[groupnames[i]] = []
	for j in range(0,len(usernames)):
		errbody[groups[j]].append(usernames[j])

	#get list of group size
	groupsizes = []
	for i in range(0,len(groupnames)):
		groupsizes.append(len(errbody[groupnames[i]]))

	#big to small
	sort_index = list(reversed(sorted(range(len(groupsizes)),key=lambda k:groupsizes[k])))

	groupinfo = np.dstack((groupnames,groupsizes))[0]
	groupinfo[np.argsort(groupinfo[:,1])]

	#big to small
	sorted_groupnames = []
	for i in range(0,len(groupnames)):
		sorted_groupnames.append(groupnames[sort_index[i]])

	max_size = len(errbody[sorted_groupnames[0]])

	return usernames,groups,names,errbody,sorted_groupnames,max_size

#create/update cost matrix
class AutoVivification(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

#if no existing cost matrix pkl in directory:
def create_matrix(usernames):
	#create new matrix with curent csv
	cost_matrix = AutoVivification()
	for i in range(len(usernames)):
		for j in range(len(usernames)):
			if usernames[i]==usernames[j]:
				cost_matrix[usernames[i]][usernames[j]] = np.inf
			elif groups[i]==groups[j]:
				cost_matrix[usernames[i]][usernames[j]] = same_group_offset
			else:
				cost_matrix[usernames[i]][usernames[j]] = 0

	return cost_matrix

def update_matrix(current_best_group,cost_matrix,usernames,groups,saved_matchups,min_unique):
	#update cost matrix
	for i in range(len(current_best_group)):
		num_names=0
		for j in range(len(current_best_group[i])):
			if isinstance(current_best_group[i][j],str):
				num_names+=1
		for j in range(num_names):
			for k in range(num_names):
				if current_best_group[i][j] != current_best_group[i][k]:
					if groups[usernames.index(current_best_group[i][j])] == groups[usernames.index(current_best_group[i][k])]:
						cost_matrix[current_best_group[i][j]][current_best_group[i][k]] += same_group_offset
					else:
						cost_matrix[current_best_group[i][j]][current_best_group[i][k]] += normal_offset
	
	#update matchup history
	new_saved_matchups = [[] for x in range(min_unique-1)]
	new_saved_matchups[0] = current_best_group
	for i in range(min_unique-2):
		new_saved_matchups[i+1] = saved_matchups[i]

	return cost_matrix,new_saved_matchups

def add_names(cost_matrix,saved_usernames,saved_groups,usernames,groups):
	for i in range(len(usernames)):
		if usernames[i] not in saved_usernames:
			#create new entry for person
			print "adding %s..." % usernames[i]
			saved_usernames.append(usernames[i])
			saved_groups.append(groups[i])

			#safety check
			if len(saved_usernames)!=len(saved_groups): import pdb; pdb.set_trace()

			#append everyone else to new person's cost_matrix entry
			#and new person to everyone else's cost_matrix entry
			for j in range(len(saved_usernames)):
				if usernames[i]==saved_usernames[j]:
					cost_matrix[usernames[i]][saved_usernames[j]] = np.inf
				elif groups[i]==saved_groups[j]:
					cost_matrix[usernames[i]][saved_usernames[j]] = same_group_offset
					cost_matrix[saved_usernames[j]][usernames[i]] = same_group_offset
				else:
					cost_matrix[usernames[i]][saved_usernames[j]] = 0
					cost_matrix[saved_usernames[j]][usernames[i]] = 0

	return cost_matrix,saved_usernames,saved_groups

def change_cost(cost_matrix,username1,username2,new_cost):
	if username1 in cost_matrix.keys() and username2 in cost_matrix.keys():
		cost_matrix[username1][username2]=new_cost
		cost_matrix[username2][username2]=new_cost
	else:
		print "uhhhhh user(s) not found..."

def find_optimal_group(current_best_group,current_best_sum,current_costs,num_per_group,usernames,cost_matrix,saved_matchups,cost_array):
	
	optimization_start = time.time()

	num_big_groups = np.mod(len(usernames),num_per_group)
	total_num_groups = int(len(usernames)/num_per_group)
	#number of tries per run
	num_tries = 100000.
	#this flag is true when a group is found where no one is paired with anyone from saved_matchups
	unique_satisifed = False
	if len(current_best_group)!=0:
		unique_satisifed = True
	
	z=0.
	#run for a minimum number of cycles or until a unique permutation is found, whichever is longer
	while z<num_tries or unique_satisifed==False:
		possible_group = [[0 for x in range(num_per_group+1)] for x in range(total_num_groups)]

		random_order = range(len(usernames))
		random.shuffle(random_order)

		counter = 0
		#lunch_groups = [{} for x in range
		for i in range(total_num_groups):
			if i<num_big_groups:
				for j in range (num_per_group+1):
					possible_group[i][j] = usernames[random_order[counter]]
					counter+=1
			else:
				for j in range (num_per_group):
					possible_group[i][j] = usernames[random_order[counter]]
					counter+=1

		#find costs
		def choose(n,k):
			return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))

		def find_cost(possible_group):
			group_costs = [0 for x in range(total_num_groups)]
			for i in range(total_num_groups):
				if i<num_big_groups:
					for j in range(num_per_group):
						for k in range(j+1,num_per_group+1):
							#import pdb; pdb.set_trace()
							group_costs[i] += cost_matrix[possible_group[i][j]][possible_group[i][k]]
					group_costs[i]=group_costs[i]/float(num_per_group+1)
				else:
					for j in range(num_per_group-1):
						for k in range(j+1,num_per_group):
							group_costs[i] += cost_matrix[possible_group[i][j]][possible_group[i][k]]
					group_costs[i]=group_costs[i]/float(num_per_group)

			return group_costs
				
		possible_group_costs = find_cost(possible_group)
		cost_sum = sum(possible_group_costs)
		#print cost_sum
		cost_array.append(cost_sum)

		def all_groups_unique(possible_group,saved_matchups):
			for i in range(len(possible_group)):
				for j in range(len(possible_group[i])):
					#import pdb; pdb.set_trace()
					if isinstance(possible_group[i][j],str):
						for k in range(len(saved_matchups)):
							for l in range(len(saved_matchups[k])):
								#import pdb; pdb.set_trace()
								if possible_group[i][j] in saved_matchups[k][l]:
									group_new = [x for x in possible_group[i] if x!=0]
									group_old = [x for x in saved_matchups[k][l] if x!=0]
									combined_group = group_new+group_old
									#import pdb; pdb.set_trace()
									if len(combined_group)-1!=len(set(combined_group)):
										#import pdb; pdb.set_trace()
										return False
			return True


		#check if the current sum is lower than the current best sum
		if cost_sum < current_best_sum:
			#check if everyone is in a unique group from 
			#import pdb; pdb.set_trace()
			if all_groups_unique(possible_group,saved_matchups):
				#import pdb; pdb.set_trace()
				unique_satisifed = True
				#then update current best variables
				current_best_group = possible_group
				current_best_sum = cost_sum
				current_costs = possible_group_costs

		z+=1

		current_elapsed = time.time()-optimization_start

		if (z/(num_tries))<=1:
			if len(current_best_group)==0:
				sys.stdout.write("\r%.2f%% completed, %d/%d permutations, elapsed: %.2fs" % (z/(num_tries)*100,z,num_tries,current_elapsed))
				sys.stdout.flush()
			else:
				sys.stdout.write("\r%.2f%% completed, %d/%d permutations, elapsed: %.2fs, possible group found" % (z/(num_tries)*100,z,num_tries,current_elapsed))
				sys.stdout.flush()
		else:
			if len(current_best_group)==0:
				sys.stdout.write("\r%.2f%% completed (overtime), %d/%d permutations, elapsed: %.2fs" % (z/(num_tries)*100,z,num_tries,current_elapsed))
				sys.stdout.flush()
			else:
				sys.stdout.write("\r%.2f%% completed (overtime), %d/%d permutations, elapsed: %.2fs, possible group found" % (z/(num_tries)*100,z,num_tries,current_elapsed))
				sys.stdout.flush()

		#num_cycles=1
		#if z>num_cycles*num_tries:
		#	print "\nno solution found in %d tries, go debug" % (num_cycles*num_tries)
		#	sys.exit(0)

	return current_best_group, current_best_sum, current_costs, cost_array

def send_email(TO,SUBJECT=None,MSG=None,ATTACH_IMG=None):
    # Import smtplib for the actual sending function
    import smtplib
    import datetime

    # Here are the email package modules we'll need
    from email.mime.image import MIMEImage
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    gmail_user = "" # email to send from
    gmail_pwd = "" # email pwd
    ME = '' # same email

    # Create the container (outer) email message.
    msg = MIMEMultipart()
    msg['Subject'] = SUBJECT
    msg['From'] = ME
    msg['To'] = ', '.join(TO)

    ## attach text message
    msg.attach(MIMEText(MSG))

    # Assume we know that the image files are all in PNG format
    if ATTACH_IMG is not None:
	    for file in ATTACH_IMG:
	        # Open the files in binary mode.  Let the MIMEImage class automatically
	        # guess the specific image type.
	        fp = open(file, 'rb')

	        img = MIMEImage(fp.read())
	        fp.close()
	        msg.attach(img)

    server = smtplib.SMTP("smtp.gmail.com", 587) # port 465 doesn't seem to work
    server.ehlo()
    server.starttls()
    server.login(gmail_user, gmail_pwd)
    server.sendmail(ME, TO, msg.as_string())
    server.close()
    return

######################################################
################ MAINNNNNN SHIITTTTTT ################
######################################################

#parse csv of responses
usernames,groups,names,errbody,sorted_groupnames,max_size = parse_responses(os.path.join(save_dir,'responses.csv'))

#load in cost matrix and history if it exists
data_path = os.path.join(save_dir,'lottery_info.pkl')
data_exist = os.path.isfile(data_path)

corrupt_flag = False
#if data exists, load in the data and update names, otherwise create new one
if data_exist:
	print '%s file detected' % (data_path)
	try:
		with open(data_path) as f:
			cost_matrix, saved_usernames, saved_groups, saved_matchups = pickle.load(f)
		#update names - if there is a name in the csv not in the matrix, add it in
		cost_matrix, saved_usernames, saved_groups = add_names(cost_matrix,saved_usernames,saved_groups,usernames,groups)
	except:
		corrupt_flag=True

if (not data_exist) or corrupt_flag:
	if not data_exist:
		print 'no lottery_info.pkl file detected, proceeding to create new cost_matrix'
	else:
		print 'saved history corrupted, proceeding to create new data'
	continue_new = 'lmao'
	while continue_new != 'c' and continue_new != 'q':
		continue_new = raw_input('(c)ontinue creating new or (q)uit?\t')
	if continue_new=='c':
		cost_matrix = create_matrix(usernames)
		saved_usernames = usernames
		saved_groups = groups
		saved_matchups = [[] for x in range(min_unique-1)]
	if continue_new== 'q':
		sys.exit(0)

#instantiate best values
current_best_group=[]
current_best_sum=np.inf
current_costs=np.inf
cost_array=[]

t=time.time()
#initial group generation
current_best_group, current_best_sum, current_costs, cost_array = find_optimal_group(current_best_group,current_best_sum,current_costs,num_per_group,usernames,cost_matrix,saved_matchups,cost_array)
print '\n'
for i in range(len(current_best_group)):
	print str(current_best_group[i])+'\tcost = '+str(current_costs[i])
print 'cost:' + str(current_best_sum)
print 'time elapsed: %g' % (time.time()-t)

#offer option to regenerate group for possible lower cost
finalize='poop'
while True:
	while finalize != 'p' and finalize != 'c' and finalize != 'v':
		finalize = raw_input('\n(p)roceed to publish, (c)ontinue group generation, or (v)iew past groups?\t')

	if finalize == 'p':
		break

	if finalize == 'c':
		finalize = 'poop'
		t=time.time()
		current_best_group, current_best_sum, current_costs, cost_array = find_optimal_group(current_best_group,current_best_sum,current_costs,num_per_group,usernames,cost_matrix,saved_matchups,cost_array)
		print '\n'
		for i in range(len(current_best_group)):
			print str(current_best_group[i])+'\tcost = '+str(current_costs[i])
		print 'cost:' + str(current_best_sum)
		print 'time elapsed: %g' % (time.time()-t)

	if finalize == 'v':
		finalize='poop'
		print '\nPrevious Groups:'
		for i in range(len(saved_matchups)):
			print ''
			for j in range(len(saved_matchups[i])):
				print saved_matchups[i][j]

#offer option to continue to save and send emails
save = 'turd'
while save != 'y' and save !='n':
	save = raw_input('Save this run and send emails? (y/n)\t')
if save == 'y':
	print '\nupdating and saving cost matrix and shit...'
	#save new variables
	cost_matrix,new_saved_matchups = update_matrix(current_best_group,cost_matrix,usernames,groups,saved_matchups,min_unique)

	with open(os.path.join(save_dir,'lottery_info.pkl'), 'w') as f:
		pickle.dump([cost_matrix,saved_usernames,saved_groups,new_saved_matchups], f)

	print 'sending emails...'

	email_me_only = False
	my_msg = ''
	for lunchgroup in current_best_group:
		email_list=[]
		name_list=[]
		for i in range(num_per_group+1):
			if isinstance(lunchgroup[i],str):
				email_list.append(lunchgroup[i]+'@mybasis.com')
				name_list.append(names[usernames.index(lunchgroup[i])]+' from '+groups[usernames.index(lunchgroup[i])])

		MSG = 'Thanks for participating in Lunch Lottery! Your group is listed below:\n' + '\n'.join(name_list) + '\n\nPlease reply to acong@mybasis.com with questions or comments (or to say hi)!'
		if email_me_only==False:
			send_email(TO=email_list, SUBJECT = "Lunch Lottery", MSG=MSG)
		
		print 'To', email_list
		print 'Message' , MSG
		my_msg=my_msg+MSG+'\n\n'

	#email me summary of this run
	#cost array trend
	plt.plot(cost_array)
	plt.axhline(y=min(cost_array),color='r',ls='dashed')
	plt.axvline(x=cost_array.index(min(cost_array)),color='r',ls='dotted')
	x1,x2,y1,y2=plt.axis()
	plt.axis((x1,x2,0,y2))
	title='Cost Trend'
	plt.title(title)
	plt.savefig(os.path.join(save_dir,'cost_trend.png'))
	plt.clf()

	#min cost vs tries
	cost_min_array=[]
	current_min = cost_array[0]
	for x in range(len(cost_array)):
		cost_min_array.append(np.min([cost_array[x],current_min]))
		if cost_array[x] < current_min:
			current_min = cost_array[x]
	plt.plot(cost_min_array)
	title='Cost Minimum'
	plt.title(title)
	plt.savefig(os.path.join(save_dir,'cost_min.png'))
	plt.clf()
	
	#cost histogram
	ca = pd.Series(np.array(cost_array),name='Cost')
	sns.set(context="paper", font="monospace")
	ax = sns.distplot(ca,bins=15,kde_kws={"color": "r", "lw": 3, "label": "KDE"})
	title='Cost Histogram'
	plt.title(title)
	plt.savefig(os.path.join(save_dir,'cost_histogram.png'))
	plt.clf()

	#cost matrix heatmap
	cost_matrix_df = pd.DataFrame.from_dict(cost_matrix)
	sns.set(context="paper", font="monospace")
	f, ax = plt.subplots(figsize=(12, 9))
	temp_df=cost_matrix_df.replace([np.inf,-np.inf],np.nan)
	minval = temp_df.min().min()
	#multiply maximum non-inf value by 1.25 to create pseudo ceiling for inf values
	maxval = temp_df.max().max()*1.25
	sns.heatmap(cost_matrix_df, vmin=minval, vmax=maxval, square=True)
	f.tight_layout()
	title='Cost Matrix'
	plt.title(title)
	plt.savefig(os.path.join(save_dir,'cost_matrix.png'))
	plt.clf()

	#cost matrix corr (Pearson correlation)
	corrmat = cost_matrix_df.corr()
	sns.set(context="paper", font="monospace")
	f, ax = plt.subplots(figsize=(12, 9))
	sns.heatmap(corrmat, vmax=1, square=True)
	f.tight_layout()
	title='Cost Matrix Correlaton'
	plt.title(title)
	plt.savefig(os.path.join(save_dir,'cost_matrix_corr.png'))
	plt.clf()

	fig_path_ct = os.path.join(save_dir,'cost_trend.png')
	fig_path_cm = os.path.join(save_dir,'cost_min.png')
	fig_path_ch = os.path.join(save_dir,'cost_histogram.png')
	fig_path_cx = os.path.join(save_dir,'cost_matrix.png')
	fig_path_cr = os.path.join(save_dir,'cost_matrix_corr.png')
	send_email(TO=['acong@mybasis.com'], SUBJECT = 'Lunch Lottery Summary', MSG=my_msg, ATTACH_IMG=[fig_path_ct,fig_path_cm,fig_path_ch,fig_path_cx,fig_path_cr])
