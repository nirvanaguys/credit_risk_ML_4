import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon="💳",
    layout="wide"
)

# ------------------------------------------------
# CSS
# ------------------------------------------------

st.markdown("""
<style>

.block-container{
padding-top:2rem;
}

.risk-box{
padding:20px;
border-radius:10px;
text-align:center;
font-size:25px;
font-weight:bold;
}

</style>
""",unsafe_allow_html=True)


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("credit_model.pkl")

    return model


model = load_model()


# ------------------------------------------------
# TITLE
# ------------------------------------------------

st.title("💳 Credit Risk Prediction Dashboard")

st.markdown(
"Predict the probability of customer default."
)


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.header("Customer Information")


rev_util = st.sidebar.slider(
"Revolving Utilization (%)",
0,
100,
30
)


age = st.sidebar.number_input(
"Age",
18,
100,
35
)


late_30_59 = st.sidebar.number_input(
"30-59 Days Late",
0,
20,
0
)


debt_ratio = st.sidebar.number_input(
"Debt Ratio",
0.0,
10.0,
0.5
)


monthly_inc = st.sidebar.number_input(
"Monthly Income",
0,
100000,
5000
)


open_credit = st.sidebar.number_input(
"Open Credit Lines",
0,
50,
5
)


late_90 = st.sidebar.number_input(
"90 Days Late",
0,
20,
0
)


real_estate = st.sidebar.number_input(
"Real Estate Loans",
0,
20,
1
)


late_60_89 = st.sidebar.number_input(
"60-89 Days Late",
0,
20,
0
)


dependents = st.sidebar.number_input(
"Dependents",
0,
10,
1
)



# ------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------

if st.sidebar.button("Predict Risk"):



    data = pd.DataFrame({

        'rev_util':[rev_util],

        'age':[age],

        'late_30_59':[late_30_59],

        'debt_ratio':[debt_ratio],

        'monthly_inc':[monthly_inc],

        'open_credit':[open_credit],

        'late_90':[late_90],

        'real_estate':[real_estate],

        'late_60_89':[late_60_89],

        'dependents':[dependents]

    })



    pred = model.predict(data)[0]



    prob = model.predict_proba(data)[0][1]



    # ------------------------
    # RISK LABEL
    # ------------------------


    if prob < 0.3:

        label = "🟢 LOW RISK"

        color = "green"



    elif prob < 0.6:

        label = "🟠 MEDIUM RISK"

        color = "orange"



    else:


        label = "🔴 HIGH RISK"

        color = "red"




    # ------------------------
    # GAUGE
    # ------------------------

    fig = go.Figure(go.Indicator(

        mode="gauge+number",


        value=prob*100,


        number={'suffix':'%'},


        title={'text':'Probability of Default'},


        gauge={

            'axis':{

                'range':[0,100]

            },


            'bar':{

                'color':'royalblue'

            },


            'steps':[


                {

                    'range':[0,30],

                    'color':'green'

                },


                {

                    'range':[30,60],

                    'color':'orange'

                },


                {

                    'range':[60,100],

                    'color':'red'

                }


            ]

        }

    ))


    fig.update_layout(

        height=350

    )



    # ------------------------
    # TOP SECTION
    # ------------------------

    c1,c2 = st.columns(2)



    with c1:


        st.plotly_chart(

            fig,

            use_container_width=True

        )




    with c2:


        st.metric(

            "Probability of Default",

            f"{prob*100:.2f}%"

        )



        st.markdown(

        f"<h2 style='color:{color}'>{label}</h2>",

        unsafe_allow_html=True

        )



    # ------------------------
    # STATISTICS
    # ------------------------

    st.subheader("Customer Statistics")



    a,b,c,d = st.columns(4)



    a.metric(

        "Age",

        age

    )


    b.metric(

        "Income",

        f"${monthly_inc:,.0f}"

    )


    c.metric(

        "Open Credit",

        open_credit

    )


    d.metric(

        "Dependents",

        dependents

    )




    # ------------------------
    # EXPLANATION
    # ------------------------

    st.subheader(

        "Why This Customer Is Risky?"

    )


    explanation=[]


    if late_90>0:

        explanation.append(

            "✓ History of 90-day late payments"

        )


    if late_30_59>0:


        explanation.append(

            "✓ Multiple 30-59 day late payments"

        )


    if late_60_89>0:


        explanation.append(

            "✓ History of 60-89 day late payments"

        )


    if monthly_inc<3000:


        explanation.append(

            "✓ Lower monthly income"

        )


    if age<30:


        explanation.append(

            "✓ Younger applicants are associated with slightly higher risk"

        )



    if len(explanation)==0:


        explanation.append(

            "✓ No major risk indicators detected"

        )



    for item in explanation:

        st.write(item)




    # ------------------------
    # INPUT TABLE
    # ------------------------

    st.subheader(

        "Input Features"

    )


    st.dataframe(

        data

    )



else:


    st.info(

        "Enter customer information from the sidebar and click Predict Risk."

    )
