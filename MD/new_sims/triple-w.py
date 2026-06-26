#!/usr/bin/python
import sys

#Formatting of a .gro file. Since w omit the velocities format10 is not needed.
format7  ='%5d%5s%5s%5d%8.3f%8.3f%8.3f\n'
format10 ='%5d%5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f\n'

def splitg(l):
    '''Splits a line from a gro file in the correct fields.'''
    f,i=float,int
    return [i(l[:5]),l[5:10],l[10:15],i(l[15:20])]+[f(i) for i in l[20:].split()][:3]
    if len(l)<69:
        o=i(l[:5]),l[5:10],l[10:15],i(l[15:20]),f(l[20:28]),f(l[28:36]),f(l[36:44])
    else:
        o=i(l[:5]),l[5:10],l[10:15],i(l[15:20]),f(l[20:28]),f(l[28:36]),f(l[36:44]),f(l[44:52]),f(l[52:60]),f(l[60:68]) 
    return list(o[:7])

# Open de input file.
try:
    infile = open(sys.argv[1],'r').readlines()
except:
    print 'First command-line argument is not a proper gro file.'
    sys.exit()

# The title, number of particles and boxsize are formatted different.
title,nbeads,box = infile.pop(0),infile.pop(0),infile.pop(-1)
outlines = []

# Just write out every line, unless it is water.
# In that case write out to extra lines.
# Putting the charges are correctly obtained random positions is not
# necessary since the possitions will be destroyed immeddiatly.
while infile:
    sl = splitg(infile.pop(0))
    outlines.append(format7%tuple(sl))
    if sl[2] == '    W':
        sl[2],sl[4],sl[5],sl[6] = 'WP',sl[4]+0.06,sl[5]+0.09,sl[6]+0.09
        outlines.append(format7%tuple(sl))
        sl[2],sl[4],sl[5],sl[6] = 'WM',sl[4]+0.07,sl[5]+0.07,sl[6]+0.10
        outlines.append(format7%tuple(sl))

# Write out file.
outfile = open(sys.argv[1][:-4]+'_PW.gro','w')
outfile.write('%s%s\n'%(title,len(outlines)))
for line in outlines:
    outfile.write(line)
outfile.write(box)
outfile.close()
print 'A water polarized file (%s_PW.gro) has been created. Velocities are not preserved!'%sys.argv[1][:-4]
