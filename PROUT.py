import streamlit as st
import pandas as pd
import numpy as np

import xgboost as xgb
import plotly.express as px
import joblib

st.set_page_config(page_title="Project Outcome Predictor")

header=st.container()
dataset=st.container()
filt=st.container()
result=st.container()



with header:
    st.title("Project Outcome Predictor")

x=pd.read_csv("PPI_data_04.csv",encoding="latin-1")
x["Project Outcome"]=np.where((x["Project status"]=="Cancelled") | (x["Project status"]=="Distressed"),"Negative","Positive")

x["ContractPeriod"]=pd.to_numeric(x["ContractPeriod"],errors="coerce")



    
R=st.selectbox("Select Region",options=x["Region"].unique())
P=st.selectbox("Select Project Type",options=x["Type of PPI"].unique())
S=st.selectbox("Select Project Sector",options=x["Primary sector"].unique())
I=st.selectbox("Select Project Country Income Level",options=x["IncomeGroup"].unique())
T=st.number_input("Specify Contract Period (years)",value=10)


st.write("Probability of +ve vs. -ve outcome for similar projects (not accounting for contract period)")
fig=px.pie(x[(x["Region"]==R)&(x["Type of PPI"]==P)&(x["Primary sector"]==S)&(x["IncomeGroup"]==I)],names='Project Outcome')
st.plotly_chart(fig)

if R=="East Asia and Pacific":
    RR=[1,0,0,0,0]
elif R=="Europe and Central Asia":
    RR=[0,1,0,0,0]
elif R=="Latin America and the Carribean":
    RR=[0,0,1,0,0]
elif R=="Middle East and North Africa":
    RR=[0,0,0,1,0]
elif R=="South Asia":
    RR=[0,0,0,0,1]
else:
    RR=[0,0,0,0,0]


R1={"East Asia and Pacific":RR[0],"Europe and Central Asia":RR[1],"Latin America and the Carribean":RR[2],"Middle East and North Africa":RR[3],"South Asia":RR[4]}

R2=pd.DataFrame(R1,index=[0])
#st.write(R2)

if P=="Brownfield":
    PP=[1,0,0]
elif P=="Divestiture":
    PP=[0,1,0]
elif P=="Greenfield Project":
    PP=[0,0,1]
else:
    PP=[0,0,0]

P1={"Brownfield":PP[0],"Divestiture":PP[1],"Greenfield Project":PP[2]}

P2=pd.DataFrame(P1,index=[0])
#st.write(P2)


if I=="Low Income":
    II=[1,0]
elif I=="Lower Middle Income":
    II=[0,1]
else:
    II=[0,0]

I1={"Low income":II[0],"Lower middle income":II[1]}

I2=pd.DataFrame(I1,index=[0])
#st.write(I2)

if S=="Water and Sewerage":
    SS=[1,0,0,0]
elif S=="Energy":
    SS=[0,1,0,0]
elif S=="Information and communication technology (ICT)":
    SS=[0,0,1,0]
elif S=="Municipal Solid Waste":
    SS=[0,0,0,1]
else:
    SS=[0,0,0,0]

S1={"Water and Sewerage":SS[0],"Energy":SS[1],"Information and communication technology (ICT)":SS[2],"Municipal Solid Waste":SS[3]}

S2=pd.DataFrame(S1,index=[0])
#st.write(S2)

T1={"ContractPeriod":T}

T2=pd.DataFrame(T1,index=[0])
T2["ContractPeriod"]=(T2["ContractPeriod"]).astype(float)
#st.write(T2)

x1=pd.concat([R2,P2,S2,T2,I2],axis=1)

st.write(x1.dtypes)

#st.write(x1)
model=joblib.load('ProjectOutcomeModel_01_xgb')

pr=model.predict(x1)

if pr[0]==1:
    st.write("Project has a relatively higher chance of distress/cancellation based on historical data and predictive model")
else:
    st.write("Historical data does and model does not indicate a relatively higher chance of distress/cancellation")

#DTI=d/ai
#RBTI=d/ai
#RBTL=d/a


#if t=="36 months":
#    	x_t=0
#else:
#    	x_t=1

#x_h=[0,0,0,0,1]
#x_p=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#if h=="Rent":
#	x_h=[0,0,0,0,1]
#elif h=="Own":
#	x_h=[0,0,0,1,0]
#else: x_h=[1,0,0,0,0]


#if p=="Car":
#	x_p=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#elif p=="Debt Consolidation":
#	x_p=[0,1,0,0,0,0,0,0,0,0,0,0,0,0]
#else: x_p=[0,0,0,0,0,0,0,0,0,0,0,1,0,0]




#z=pd.DataFrame()
#z["term_ 60 months"]=[x_t]
#z["LTI"]=[DTI]
#z["RBTI"]=[RBTI]
#z["RBTL"]=[RBTL]
#st.write(z)

#hvd=pd.DataFrame({"home_ownership_MORTGAGE":[x_h[0]],"home_ownership_NONE":[x_h[1]],"home_ownership_OTHER":[x_h[2]],"home_ownership_OWN":[x_h[3]],"home_ownership_RENT":[x_h[4]]})

#pvd=pd.DataFrame({"purpose_credit_card":[x_p[0]],"purpose_debt_consolidation":[x_p[1]],"purpose_educational":[x_p[2]],"purpose_home_improvement":[x_p[3]],"purpose_home_improvement":[x_p[4]],"purpose_house":[x_p[5]],"purpose_major_purchase":[x_p[6]],"purpose_medical":[x_p[7]],"purpose_moving":[x_p[8]],"purpose_other":[x_p[9]],"purpose_renewable_energy":[x_p[10]],"purpose_small_business":[x_p[11]],"purpose_vacation":[x_p[12]],"purpose_wedding":[x_p[13]]})


#x=pd.concat([pvd,hvd,z],axis=1)

#st.write(x)

#model=joblib.load('ProjectOutcomeModel_01')

#p=model.predict(x)

#if p[0]==0:
#	st.write("Your loan application cannot be pre-approved now. Please contact the helpline for additional information")
#else: st.write("Your loan application is pre-approved. Loan Services Department will reach out to you on your phone number on file for the next #steps. Thank you!") 






