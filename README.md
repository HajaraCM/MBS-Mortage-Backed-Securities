# Mortgage-Backed Securities Prepayment Estimation Tool

## Overview

The Mortgage-Backed Securities (MBS) Prepayment Estimation Tool is a multi-model pipeline designed to build and deploy classification and regression models. This project focuses on predicting loan statuses (delinquent or non-delinquent) and prepayment amounts. Additionally, other models are developed to assess prepayment risk and evaluate monthly installment affordability.The application analyzes data from Freddie Mac to uncover trends and indicators that can assist in anticipating consumer behavior regarding mortgage prepayments

## Features

- **Exploratory Data Analysis (EDA)**
- **Data Preprocessing**
- **Feature Selection**
  - **ANOVA** for regression tasks
  - **Mutual Information Score** for classification tasks
- **Multi-Model Pipeline**: 
  - **Classification Model**: Predicts loan status (delinquent or non-delinquent).
  - **Regression Model**: Predicts prepayment amounts.
- **Additional Models**:
  - **Prepayment Risk Model**: Estimates the likelihood of prepayment.
  - **Affordability Model**: Evaluates monthly installment affordability.
- **Imbalanced Data Handling**: Utilizes oversampling techniques (e.g., SMOTE).
- **Deployment**: Deployed on Render with a user-friendly web interface.
- **User Interface**: Developed using Flask, HTML, and CSS.

## Dataset Features

1.	Credit score of the client. 
2.	The maturity date of the mortgage. 
3.	The amount or percentage of insurance on the mortgage 
4.	Debit to income ration of the borrower 
5.	Mortgage interest rate  
6.	The purpose of the loan. 
7.	Loan sequence number which denotes the unique loan ID 
8.	The number of borrowers issued on the loan. 
9.	Prepayment Penalty Mortgage which denotes if there is any penalty levied on prepayment of the loan. 
10.	The property type, the state in which property is and its postal code and address. 
11.	The information about the seller and service company. 
12.	HARP indicator-denotes if the loan is HARP or non-HARP, 
13.	Interest only indicator-Denotes if the loan requires only the interest payments over the period of maturity or not respectively. 
14.	debt-to-income ratio(DTI): Your debt-to-income ratio (DTI) compares how much you owe each month to how much you earn. Specifically, it's the percentage of your gross monthly income (before taxes) that goes towards payments for rent, mortgage, credit cards, or other debt. 
15.	PPM: A type of mortgage that requires the borrower to pay a penalty for prepayment, partial payment of principal or for repaying the entire loan within a certain time period. A partial payment is generally defined as an amount exceeding 20% of the original principal balance. 
16.	NumBOrrowers: A borrower describes an individual, entity, or organization applying for funds, i.e., a loan from a lender under an agreement to repay the same later.  
17.	Everdeliquent: Being delinquent refers to the state of being past due on a debt. Delinquency occurs as soon as a borrower misses a payment on a loan, which can affect their credit score. Delinquency rates are used to show how many accounts in a financial institution's portfolio are delinquent. 
18.	IsFirstTimeHomebuyer: According to the agency, a first-time homebuyer is: Someone who hasn't owned a principal residence for the three-year period ending on the date of purchase of the new home. An individual who has never owned a principal residence even if the person's spouse was a homeowner. 
19.	Creditrange: A credit score is a prediction of your credit behavior, such as how likely you are to pay a loan back on time, based on information from your credit reports.  
20.Monthly_income: Your gross monthly income includes all sources of money that you receive over the course of a month, including but not limited to regular wages, earnings from side jobs and investments. Lenders consider your gross monthly income to determine your creditworthiness and ability to repay loans. 
21.Prepayment: Prepayment is an accounting term for the settlement of a debt or installment loan in advance of its official due date. A prepayment may be the settlement of a bill, an operating expense, or a non-operating expense that closes an account before its due date. 
22.	MSA: MSAs (Marketing Services Agreement) with Mortgage Companies. 
23.	MIP: Federal Housing Administration (FHA) mortgage loans are designed to       help people who might have trouble getting other types of mortgage loans to buy a home. And if a homebuyer uses an FHA-backed loan, they're required to pay a mortgage insurance premium (MIP). 
24.	OCLTV: Loan-to-value (LTV) is calculated simply by taking the   loan amount and dividing it by the value of the asset or collateral being borrowed against. 
25.	debt-to-income ratio:Your debt-to-income ratio (DTI) is all your monthly debt payments  divided by your gross monthly income. This number is one way lenders measure your ability to manage the monthly payments to repay the money you plan to borrow. Different loan products and lenders will have different DTI limits. 
26.	OrigUPB :Loan origination is the process by which a borrower applies for a new loan, and a lender processes that application. Origination generally includes all the steps from taking a loan application up to disbursal of funds (or declining the application). 
  
27 .PropertyState:  Property State means, with respect to a particular parcel of Land or Security Instrument purporting to encumber it, the state where such Land is located. 
  
28.	LTV_Range:The loan-to-value (LTV) ratio is an assessment of lending risk that financial institutions and other lenders examine before approving a mortgage. Typically, loan assessments with high LTV ratios are considered higher-risk loans. Therefore, if the mortgage is approved, the loan has a higher interest rate. 
  
29.	MonthsDelinquent:A delinquent status means that you are behind in your payments. The length of time varies by lender and the type of debt, but this period generally falls anywhere between 30 to 90 days. 
 
30.	Prefix: The designation assigned by the issuer denoting the type of the loans and the security 
31.	Original Interest Rate: The interest rate of the loan as stated on the note at the time the loan was originated or modified.  
 

## Purpose

To empower stakeholders in the MBS market with reliable predictions regarding loan status, prepayment amounts, prepayment risk, and monthly installment affordability.

## Getting Started

### Prerequisites

- Python 3.x
- Required libraries (listed in `requirements.txt`)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mbs-prepayment-tool.git
cd mbs-prepayment-tool

# Install dependencies
pip install -r requirements.txt


