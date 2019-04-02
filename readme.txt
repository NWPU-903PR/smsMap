====================================================================
		    
 smsMap                               04/01/2019
 ver1.0.0                           ICI/LIFT, China
		    	 	       
====================================================================

System Requirements
===================

Software Requirements
---------------------
The smsMap is supported on the following operating systems 
   
   Linux
   Windows 7
   Windows 10 

Hardware Requirements/Recommendations
-------------------------------------
   Intel Core i3 2100 or later for Windows 
   Color display capable of 1024 X 768 pixel resolution
   

Memory Requirements/Recommendations
-------------------------------------
smsMap  requires approximately
2G of available disk space on the drive or partition.

smsMap requires a minimum of
8G of RAM for mapping long noisy SMS reads.


User's Guide
=================

Input Sequences Requirements
----------------------------
Input file requires fasta or fastq format


Execute Step
------------
Step 1: compile codes using make command
Step 2: ./locate or ./smsMap then come out options reuired


A running example
-----------
./locate --index genome.fa
./locate --search genome.fa --seq reads.fa >pos.txt
./smsMap --seq reads.fa --genome genome.fa --pos pos.txt --out smsMap_aligned.txt

The final mapping file is smsMap_aligned.txt




Copyright Notice
===================
Software, documentation and related materials:
Copyright (c) 2019-2021 
Institute of Control & Information(ICI), Northwestern Polytechnical University, China
Key Laboratory of Information Fusion Technology(LIFT), Ministry of Education, China
All rights reserved.