function [sam_files,count,token]=custom_filebyfileload(d,file_name,flag)

%reads files in a tokenized order and prepares some variables for dataset

files = dir(fullfile(d, 'sam*.csv'))%%choose all the files starting with sam
sam_files=files;
count=1;
token=[];

