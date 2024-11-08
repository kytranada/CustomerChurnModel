import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_churn_analysis_charts(data):
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Churn Distribution',
            'Churn by Internet Service',
            'Average Monthly Charges by Service',
            'Customer Tenure Distribution'
        ),
        specs=[[{'type': 'pie'}, {'type': 'pie'}],
               [{'type': 'scatter'}, {'type': 'histogram'}]]
    )

    # 1. Churn Distribution Pie Chart (Remains the same)
    churn_dist = data['Churn Value'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=['Retained', 'Churned'],
            values=churn_dist.values,
            hole=0.4,
            marker_colors=['#2ecc71', '#e74c3c']
        ),
        row=1, col=1
    )

    # 2. Churn by Internet Service as Pie Chart
    churn_by_service = data.groupby('Internet Service')['Churn Value'].mean()
    fig.add_trace(
        go.Pie(
            labels=churn_by_service.index,
            values=churn_by_service.values * 100,
            marker_colors=['#3498db', '#9b59b6', '#f1c40f'],
            name='Churn Rate'
        ),
        row=1, col=2
    )

    # 3. Average Monthly Charges by Service Type as Area Chart
    avg_charges = data.groupby('Internet Service')['Monthly Charge'].mean().sort_values()
    fig.add_trace(
        go.Scatter(
            x=avg_charges.index,
            y=avg_charges.values,
            mode='lines',
            fill='tozeroy',
            line_color='#9b59b6',
            name='Avg Monthly Charge'
        ),
        row=2, col=1
    )

    # 4. Customer Tenure Distribution (Remains the same)
    fig.add_trace(
        go.Histogram(
            x=data['Tenure in Months'],
            nbinsx=30,
            marker_color='#f1c40f',
            name='Tenure Distribution'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Customer Churn Analysis Dashboard",
        title_x=0.5,
        title_font_size=24
    )

    # Update axes labels
    # For pie charts, axes titles are not necessary
    fig.update_xaxes(title_text="Internet Service Type", row=2, col=1)
    fig.update_xaxes(title_text="Tenure (Months)", row=2, col=2)
    fig.update_yaxes(title_text="Average Monthly Charge ($)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=2)

    return fig



def create_demographic_analysis(data):
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Age Distribution by Churn Status',
            'Churned Customers by Gender',
            'Churn Rate by Streaming Service',
            'Average Monthly Charges by Age Group'
        ),
        specs=[[{'type': 'box'}, {'type': 'domain'}],
               [{'type': 'bar'}, {'type': 'scatter'}]]
    )

    # 1. Age Distribution by Churn Status (Box Plot)
    fig.add_trace(
        go.Box(
            x=data[data['Churn Value'] == 0]['Age'],
            name='Retained',
            marker_color='#2ecc71'
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(
            x=data[data['Churn Value'] == 1]['Age'],
            name='Churned',
            marker_color='#e74c3c'
        ),
        row=1, col=1
    )

    # 2. Churned Customers by Gender (Pie Chart)
    gender_churn_counts = data[data['Churn Value'] == 1]['Gender'].value_counts()
    fig.add_trace(
        go.Pie(
            labels=gender_churn_counts.index,
            values=gender_churn_counts.values,
            marker_colors=['#3498db', '#e74c3c'],
            name='Churned Customers by Gender'
        ),
        row=1, col=2
    )

    # 3. Churn Rate by Streaming Service (Horizontal Bar Chart)
    streaming_churn = data.groupby('Streaming')['Churn Value'].mean().sort_values()
    fig.add_trace(
        go.Bar(
            x=streaming_churn.values * 100,
            y=streaming_churn.index,
            orientation='h',
            marker_color='#9b59b6'
        ),
        row=2, col=1
    )

    # 4. Average Monthly Charges by Age Group (Line Chart)
    data['Age Group'] = pd.cut(
        data['Age'],
        bins=[0, 30, 45, 60, 100],
        labels=['18-30', '31-45', '46-60', '60+']
    )
    age_charges = data.groupby('Age Group')['Monthly Charge'].mean().reset_index()
    fig.add_trace(
        go.Scatter(
            x=age_charges['Age Group'],
            y=age_charges['Monthly Charge'],
            mode='lines+markers',
            line=dict(color='#f1c40f'),
            marker=dict(size=10),
            name='Avg Monthly Charge'
        ),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Customer Demographics Analysis",
        title_x=0.5,
        title_font_size=24
    )

    # Update axes labels
    fig.update_xaxes(title_text="Age", row=1, col=1)
    fig.update_xaxes(title_text="Churn Rate (%)", row=2, col=1)
    fig.update_xaxes(title_text="Age Group", row=2, col=2)

    fig.update_yaxes(title_text="Streaming Service", row=2, col=1)
    fig.update_yaxes(title_text="Average Monthly Charge ($)", row=2, col=2)

    return fig



def get_churn_insights(data):
    insights = {
        'avg_tenure_churned': data[data['Churn Value'] == 1]['Tenure in Months'].mean(),
        'avg_tenure_retained': data[data['Churn Value'] == 0]['Tenure in Months'].mean(),
        'avg_charge_churned': data[data['Churn Value'] == 1]['Monthly Charge'].mean(),
        'avg_charge_retained': data[data['Churn Value'] == 0]['Monthly Charge'].mean(),
        'median_age_churned': data[data['Churn Value'] == 1]['Age'].median(),
        'median_age_retained': data[data['Churn Value'] == 0]['Age'].median(),
        'gender_churn_rates': data.groupby('Gender')['Churn Value'].mean() * 100
    }
    return insights

def create_churn_by_age_and_gender(data):
    # Create a new 'Age Group' column
    data['Age Group'] = pd.cut(data['Age'], bins=[0, 30, 45, 60, 100], labels=['18-30', '31-45', '46-60', '60+'])

    # Calculate churn rates by age group and gender
    churn_by_age_gender = data.groupby(['Age Group', 'Gender'])['Churn Value'].mean().reset_index()
    churn_by_age_gender['Churn Rate'] = (churn_by_age_gender['Churn Value'] * 100).round(2)

    # Create the visualization
    fig = go.Figure(data=[
        go.Bar(
            x=churn_by_age_gender[(churn_by_age_gender['Gender'] == 'Male')]['Age Group'],
            y=churn_by_age_gender[(churn_by_age_gender['Gender'] == 'Male')]['Churn Rate'],
            name='Male'
        ),
        go.Bar(
            x=churn_by_age_gender[(churn_by_age_gender['Gender'] == 'Female')]['Age Group'],
            y=churn_by_age_gender[(churn_by_age_gender['Gender'] == 'Female')]['Churn Rate'],
            name='Female'
        )
    ])

    fig.update_layout(
        barmode='group',
        xaxis_title='Age Group',
        yaxis_title='Churn Rate (%)',
        title='Churn Rate by Age Group and Gender',
        title_x=0.5
    )

    return fig

def create_churn_heatmap_by_region(data):
    # Calculate churn rates by state
    churn_by_state = data.groupby('State')['Churn Value'].mean().reset_index()
    churn_by_state['Churn Rate'] = (churn_by_state['Churn Value'] * 100).round(2)

    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        x=churn_by_state['State'],
        y=[''],
        z=churn_by_state['Churn Rate'],
        colorscale='Blues',
        colorbar_title='Churn Rate (%)'
    ))

    fig.update_layout(
        title='Churn Rate by State',
        title_x=0.5,
        xaxis_title='State',
        yaxis_title=''
    )

    return fig

def create_total_charges_vs_churn(data):
    # Create the scatter plot
    fig = go.Figure(data=go.Scatter(
        x=data['Total Charges'],
        y=data['Churn Value'],
        mode='markers',
        marker=dict(
            color=data['Churn Value'],
            colorscale='Viridis',
            size=7
        )
    ))

    fig.update_layout(
        title='Total Charges vs Churn',
        title_x=0.5,
        xaxis_title='Total Charges ($)',
        yaxis_title='Churn Value (0=No, 1=Yes)'
    )

    return fig