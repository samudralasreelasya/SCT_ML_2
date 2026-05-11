# Customer Segmentation using K-Means Clustering

## Overview
This project focuses on customer segmentation for a retail store using the K-Means Clustering algorithm. The goal is to group customers based on their annual income and spending behavior. This grouping helps businesses understand different customer categories.

K-Means is an unsupervised machine learning algorithm used to find hidden patterns or groups in data.

---

## Objective
The main aims of this project are:
- Analyze customer purchasing behavior
- Group customers into various clusters
- Visualize customer segments
- Help businesses improve marketing strategies

---

## Dataset
The dataset has the following features:

| Column Name | Description |
|---|---|
| CustomerID | Unique customer ID |
| Gender | Male/Female |
| Age | Customer age |
| Annual Income (k$) | Annual income in thousand dollars |
| Spending Score (1-100) | Customer spending score assigned by the store |

For clustering, the following features were used:
- Annual Income (k$)
- Spending Score (1-100)

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Machine Learning Algorithm
### K-Means Clustering
K-Means clustering divides customers into groups based on similarities in their spending patterns.

---

## Project Workflow

1. Import required libraries
2. Load the dataset
3. Perform data analysis
4. Select important features
5. Use the Elbow Method to find the optimal number of clusters
6. Apply K-Means clustering
7. Visualize customer segments
8. Interpret the clusters

---

## Elbow Method
The Elbow Method helps determine the best number of clusters for the dataset by calculating WCSS (Within Cluster Sum of Squares).

---

## Output
The project generates:
- Elbow Method graph
- Customer Segmentation graph
- Clustered customer groups

---

## Customer Segments
The model identifies different types of customers, including:
- High income & high spending customers
- High income & low spending customers
- Low income & high spending customers
- Low income & low spending customers
- Average customers

---

## Conclusion
The K-Means clustering model successfully grouped customers into meaningful segments based on income and spending score. These insights can assist retail businesses in enhancing customer targeting and marketing strategies.

---

## How to Run the Project

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```