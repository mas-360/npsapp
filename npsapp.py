# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 13:08:48 2023

@author: 27823
"""
from PIL import Image
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards


st.set_page_config(page_title="Coconut Cosmetics CX",
                   layout="wide", initial_sidebar_state="expanded"
)
#---logo
img_logo = Image.open("images/Capture.PNG")


#---SIDEBAR---
st.sidebar.header("Customer Experience Dashboard")
st.sidebar.markdown("Net Promoter Score (NPS), the ultimate loyalty metric, empowers you to gauge customer devotion and benchmark your performance in the market."
                    )
st.sidebar.markdown("---")
st.sidebar.markdown(""" **Customer Breakdown:**  
                    **Detractors -** Are unhappy customers who can damage your brand and impede growth through negative word of mouth.  
                    **Passives -** Are satisfied but unethusiastic customers who are vulnerable to competitive offerings.  
                    **Promotors -** Are loyal enthusiasts who will keep buying and fuel growth by referring to others. 
                    """)
st.sidebar.markdown("---")                    
st.sidebar.markdown("""**Insight Gained:**  
                    Improving your NPS means converting your passives and detractors into promoters with your efforts and keeping your promoters happy.
                    """)
 
#st.title("Customer Experience Dashboard")
with st.container():
    image_col, txt_col = st.columns(2)
    with image_col:
        st.image(img_logo)
    with txt_col:
        st.empty()
        
st.markdown("---")

#TOP KPI'S
st.subheader("Top KPI's:")

def style_metric_cards(
    background_color: str = "#FFF",
    border_size_px: int = 1,
    border_color: str = "#CCC",
    border_radius_px: int = 5,
    border_left_color: str = "#62929e",
    box_shadow: bool = True,
):

    box_shadow_str = (
        "box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15) !important;"
        if box_shadow
        else "box-shadow: none !important;"
    )
    st.markdown(
        f"""
        <style>
            div[data-testid="metric-container"] {{
                background-color: {background_color};
                border: {border_size_px}px solid {border_color};
                padding: 5% 5% 5% 10%;
                border-radius: {border_radius_px}px;
                border-left: 0.5rem solid {border_left_color} !important;
                {box_shadow_str}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
col1, col2, col3 = st.columns(3)
col1.metric(label="Total Customers:", value="3,594", delta=132)
col2.metric(label="Customer Acquisition Cost:", value="$22", delta="-10%")
col3.metric(label="Average Churn Rate:", value="27%", delta="0%")
style_metric_cards()
st.markdown("---")  

st.subheader("NPS Survey Results:") 
#HORIZONTAL BAR CHART - nps by company Feb 2020
data = {"Industry_Competitor": ["Competitor A", "Competitor B", "Competitor C", "OUR BUSINESS", "Competitor D", "Competitor E", "Competitor F", "Competitor G", "Competitor H", "Competitor I", "Competitor J",
                                "Competitor K", "Competitor L", "Competitor M", "Competitor N", "Competitor O"],
        "NPS": [47, 44, 40, 37, 32, 30, 29, 28, 28, 28, 27, 24, 21, 21, 20, 18]}

df2 = pd.DataFrame.from_dict(data)

df2 = df2.sort_values(by="NPS", ascending=True)
df2['color'] = df2['Industry_Competitor'].apply(lambda x: "#62929e" if x == "OUR BUSINESS" else "#BFC0C0")
y = df2["Industry_Competitor"]
fig = go.Figure(go.Bar(
    x=df2["NPS"],
    y=df2["Industry_Competitor"],
    marker_color=df2['color'],  # Use the 'color' column to determine the color of each bar
    orientation="h",
    text=df2["NPS"],
    textposition="auto",
    textfont=dict(
        family='Arial',
        size=14,
        color='rgb(82, 82, 82)',
    ),
))

fig.update_layout(
    xaxis=dict(
            title="NPS",
            titlefont=dict(
                family='Arial',
                size=14,
                color='rgb(82, 82, 82)',
            ),
            showgrid=False,
            showticklabels=True,
            showline=True,
            linecolor='black',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=14,
            ),
        ),
    yaxis=dict(
        
        showgrid=False,
        showline=False,
        zeroline=False,
        linecolor='whitesmoke',
        linewidth=1,
        tickfont=dict(
            family='Arial',  # Set the y-axis font to Arial
            size=14),
    ),
        margin=dict(l=140, r=40, b=50, t=80),
        legend=dict(
            font_size=10,
            yanchor='middle',
            xanchor='right',
        ),
        width=800,
        height=600,
        paper_bgcolor='whitesmoke',
        plot_bgcolor='whitesmoke',
        hovermode='closest',
    )
annotations = []
# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='NPS: we rank 4th among competitors - February 2023.',
                              font=dict(family='Arial',
                                        size=25,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig.update_layout(annotations=annotations) 
    
#STACKED BAR CHART - nps overtime

colors = ["#62929e", "#BFC0C0", "#BFC0C0", "#BFC0C0"]
labels = ["OUR BUSINESS", "Competitor A", "Competitor B", "Competitor C", "Competitor D"]
mode_size = [12, 8, 8, 8, 8]
line_size = [4, 2, 2, 2, 2]

x_data = np.arange(1, 14)
y_data = np.array([
    [29,29,31,34,34,34,32,34,35,35,33,34,35,37],
    [49,48,48,47,45,46,45,45,44,43,42,42,44,47],
    [38,39,39,38,40,40,40,39,38,39,39,40,41,44],
    [36,37,38,37,38,39,38,37,40,38,36,37,38,40],
    [25,25,26,24,25,26,27,26,26,28,28,29,32,32],    
])

fig1 = go.Figure()
for i in range(min(len(colors), len(labels), len(mode_size), len(line_size))):
    fig1.add_trace(go.Scatter(
        x=x_data,
        y=y_data[i],
        name=labels[i],
        mode='lines+markers',
        marker=dict(color=colors[i], size=mode_size[i]),
        line=dict(color=colors[i], width=line_size[i])
    ))

fig1.update_layout(
    xaxis=dict(
        title = "Month",
        titlefont=dict(
            family='Arial',
            size=14,
            color='rgb(82, 82, 82)',
        ),
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='black',
        linewidth=1,
        dtick=1,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=14,
            color='rgb(82, 82, 82)',
        ),
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=False,
        showticklabels=False,
    ),
    autosize=False,
    margin=dict(
        autoexpand=False,
        l=100,
        r=20,
        t=110,
    ),
    showlegend=False,
    height=600,
    paper_bgcolor='whitesmoke',
    plot_bgcolor='whitesmoke',
)

annotations = []

# Adding labels
for y_trace, label, color in zip(y_data, labels, colors):
    # labeling the left_side of the plot
    annotations.append(dict(xref='paper', x=0.05, y=y_trace[0],
                            xanchor='right', yanchor='middle',
                            text=label +' '+ str(y_trace[0]),
                            font=dict(family='Arial', size=14),
                            showarrow=False))
    # labeling the right_side of the plot
    annotations.append(dict(xref='paper', x=0.95, y=y_trace[13],  # Use index 13 instead of 14
                            xanchor='left', yanchor='auto',
                            text=str(y_trace[13]),  # Convert the value to a string
                            font=dict(family='Arial', size=14),
                            showarrow=False))
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.10,
                                  xanchor='left', yanchor='bottom',
                                  text='Great Work! Continued progress towards improving NPS.',
                                  font=dict(family='Arial',
                                            size=25,
                                            color='rgb(37,37,37)'),
                                  showarrow=False))

fig1.update_layout(annotations=annotations)


left_col, right_col = st.columns(2)
left_col.plotly_chart(fig1, theme=None, use_container_width=True)
right_col.plotly_chart(fig, theme=None, use_container_width=True)

#HORIZONTAL BAR GRAPHS - nps comment
top_labels = ["Comment", "No Comment"]
colors = ["#BFC0C0", "#e5e5e5"]
d_colors = ["#EF8354","#f9c784"]
x_data = [[15, 85],
           [10, 90],
           [16, 84],
           [29, 71]]
total_data = [[7195, 3482, 2624, 1089]]
y_data = ["Total", "Promoters", "Passives", "Detractors"]

fig = go.Figure()
for i in range(0, len(x_data[0])):
    for xd, yd in zip(x_data, y_data):
        if yd == "Detractors":
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=d_colors[i],  # Set the color to "#EF8354" for "Detractors" bar
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))
        else:
            fig.add_trace(go.Bar(
                x=[xd[i]], y=[yd],
                orientation='h',
                marker=dict(
                    color=colors[i],
                    line=dict(color='rgb(248, 248, 249)', width=1)
                )
            ))

fig.update_layout(   
    xaxis=dict(
        title="% of Total",
        showgrid=False,
        showline=False,
        showticklabels=False,
        zeroline=False,
        domain=[0.15, 1]
    ),
    yaxis=dict(        
        showgrid=False,
        showline=True,
        showticklabels=False,
        zeroline=False,
    ),
    barmode='stack',
    paper_bgcolor='whitesmoke',
    plot_bgcolor='whitesmoke',
    margin=dict(l=100, r=10, t=140, b=80),
    showlegend=False,
    height=600,
)

annotations = []

for yd, xd in zip(y_data, x_data):
    # labeling the y-axis
    annotations.append(dict(xref='paper', yref='y',
                            x=0.14, y=yd,
                            xanchor='right',
                            text=str(yd),
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False, align='right'))
    # labeling the first percentage of each bar (x_axis)
    annotations.append(dict(xref='x', yref='y',
                            x=xd[0] / 2, y=yd,
                            text=str(xd[0]) + '%',
                            font=dict(family='Arial', size=14,
                                      color='rgb(67, 67, 67)'),
                            showarrow=False))
    # labeling the first Likert scale (on the top)
    if yd == y_data[-1]:
        annotations.append(dict(xref='x', yref='paper',
                                x=xd[0] / 2, y=1.1,
                                text=top_labels[0],
                                font=dict(family='Arial', size=14,
                                          color='rgb(67, 67, 67)'),
                                showarrow=False))
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.20,
                                  xanchor='left', yanchor='bottom',
                                  text='NPS Comments: Detractors more vocal with product feedback.',
                                  font=dict(family='Arial',
                                            size=25,
                                            color='rgb(37,37,37)'),
                                  showarrow=False))
    space = xd[0]
    for i in range(1, len(xd)):
            # labeling the rest of percentages for each bar (x_axis)
            annotations.append(dict(xref='x', yref='y',
                                    x=space + (xd[i]/2), y=yd,
                                    text=str(xd[i]) + '%',
                                    font=dict(family='Arial', size=14,
                                              color="rgb(67, 67, 67)"),
                                    showarrow=False))
            # labeling the Likert scale
            if yd == y_data[-1]:
                annotations.append(dict(xref='x', yref='paper',
                                        x=space + (xd[i]/2), y=1.1,
                                        text=top_labels[i],
                                        font=dict(family='Arial', size=14,
                                                  color='rgb(67, 67, 67)'),
                                        showarrow=False))
            space += xd[i]
            
fig.update_layout(annotations=annotations)


#STACKED BAR CHART - nps component breakdown

labels = ["Detractor", "Passive", "Promotor"]
colors = ["#EF8354", "#BFC0C0", "#e5e5e5"]
y_data2 = [[10, 11, 12, 10, 11, 12, 14, 13, 15, 17, 20, 21, 22, 25],
           [51, 49, 45, 46, 44, 42, 40, 40, 35, 31, 27, 24, 21, 13],
           [39, 40, 43, 44, 45, 46, 46, 47, 50, 52, 53, 55, 57, 62],
]
x_data2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

fig3 = go.Figure()

for i in range(len(y_data2)):
    fig3.add_trace(go.Bar(
        x=x_data2,
        y=y_data2[i],
        marker=dict(
            color=colors[i],
            line=dict(color='rgb(248, 248, 249)', width=1)
        ),
        name=labels[i],
        text=y_data2[i],
        textposition="auto",
        textfont=dict(
            family='Arial',
            size=14,
            color='rgb(82, 82, 82)',
        ),
    ))

fig3.update_layout(
    xaxis=dict(
        title="Month",
        titlefont=dict(
            family='Arial',
            size=14,
            color='rgb(82, 82, 82)',
        ),
        showgrid=False,
        showline=True,
        showticklabels=True,
        zeroline=True,
        linecolor='black',
        linewidth=1,
        dtick=1,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=14,
            color='black',
        ),
        #domain=[0.15, 1]
    ),
    yaxis=dict(
        title="% of Total",
        titlefont=dict(
            family='Arial',
            size=14,
            color='rgb(82, 82, 82)',
        ),
        showgrid=False,
        showline=True,
        showticklabels=True,
        zeroline=False,
        linecolor='black',
        linewidth=1,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=14,
            color='black',
        ),
    ),
    barmode='stack',
    paper_bgcolor='whitesmoke',
    plot_bgcolor='whitesmoke',
    margin=dict(l=140, r=40, b=50, t=80),
    height=600,
    #margin=dict(l=120, r=10, t=140, b=80),
    showlegend=True,
    legend=dict(
        x=0,
        y=1.05,
        xanchor="left",
        yanchor="top",
        orientation="h",
        font=dict(
            family='Arial', size=14)
))
annotations = []
# Title
annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='NPS over time: Detractors increasing, Passives shrink.',
                              font=dict(family='Arial',
                                        size=25,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig3.update_layout(annotations=annotations)

left_col1, right_col1 = st.columns(2)
left_col1.plotly_chart(fig3, theme=None, use_container_width=True)
right_col1.plotly_chart(fig, theme=None, use_container_width=True)
