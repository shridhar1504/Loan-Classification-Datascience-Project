# Loan-Classification-Datascience-Project
This project uses machine learning algorithms to predict the classification of loan status. The dataset is loaded and some transformation is done using SQL for getting a proper dataset with some valid informations. The project uses a loan applications which includes information about the individuals who are applying for the loan. With that data, training the model and getting the high accuracy model and using it predicting the future loan status with the needed data.
## Problem Statement :
Loan Lenders need to be able to accurately predict whether the loan will be repaid inorder to minimize the risk. By predicting the loan status of every individual, the lenders can able to follow the results of it to find out the differences among the behaviourial pattern of the customer. This is challenging problem because there are many factors that can influence the loan status.
## Solution Approach :
With the help of Structured Query Language (SQL), transforming the data with the bunch of datasets which has different informations on Customer's behaviourial pattern. By the following structural mapping, joining the datasets to get the final data.  


`image` ![Dataset SQL Formatting](https://github.com/shridhar1504/Loan-Classification-Datascience-Project/assets/113985416/984a2561-6d2f-4656-bfe2-e47f970c175d)


Machine Learning can be used to build models that can predict the loan status such as A, B, C & D (i.e., Approved, Pending, Denied, Closed). These models are trained on historical data of loan applications and using this data which learns with the relationship between factors that impacts on status of the loan. Once a model is trained, it can be used to predict the loan status of the future applicants.
## Observations :
The accuracy of the loan classification model can vary depending on the dataset that is used to train the model. The Bank had given applicant's details in individual CSV Files. For the predictive modeling,by using SQL; the datasets should be joined or merged using various statements which can have all the necessary column to form the final data in a csv format. The individual datasets and details of the datasets are as follows :
    * Account - The dataset has account id, district id, frequency & date.
    * Card - The dataset has card id, disposition id, type & issued.
    * Client - The dataset has client id , birth number alomg with district id.
    * Disp - The dataset has disposition id, client id, account id, type.
    * District - The dataset has different factors such as A1 - A16 which includes much datas than the other datasets but not explained in a well manner.
    * Loan - The dataset has loan id, account id, date, amount, duartion, payment & status.
    * Order - The dataset has order id, account id, bank to, account to, amount, k_symbol.
    * Transaction Data - The dataset has transaction id, account id, date, type, operation, amount, balance, k_symbol, bank & account.
