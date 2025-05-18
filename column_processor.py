import pandas as pd
import numpy as np
from datetime import datetime

def process_single_column(col_data, col_name, total_records, primary_keys=None, original_df=None):
    """Process a single column and return its insights."""
    try:
        insights = {
            'column_name': col_name,
            'charts': [],
            'tables': [],
            'text_content': []
        }
        
        col_data = col_data.dropna()
        coverage = 100 - (col_data.isnull().sum() / total_records * 100)
        coverage = max(0, min(100, float(coverage) if pd.notna(coverage) else 0))
        
        insights['text_content'].append(f"Coverage: {coverage:.2f}%")
        nunique = col_data.nunique()
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        # Initialize val_counts_total for all cases
        top_n = 10
        val_counts_total = col_data.value_counts().head(top_n)
        percent_total = (val_counts_total / total_records * 100).round(2)

        if nunique == total_records:
            value_counts = col_data.value_counts().reset_index()
            value_counts.columns = ["Value", "Count"]
            value_counts = value_counts.sort_values("Count", ascending=False)
            insights['tables'].append(('Unique Values', value_counts))
        elif not is_numeric:
            try:
                # Try parsing as datetime
                parsed_col = pd.to_datetime(col_data, errors="coerce", infer_datetime_format=True)
                if parsed_col.notna().sum() == 0 and pd.api.types.is_numeric_dtype(col_data):
                    parsed_col = pd.to_datetime(col_data, errors="coerce", unit="ms")
                    if parsed_col.notna().sum() == 0:
                        parsed_col = pd.to_datetime(col_data, errors="coerce", unit="s")

                if parsed_col.notna().sum() > 0:
                    insights['is_datetime'] = True
                    insights['parsed_datetime'] = parsed_col
                    return insights
            except Exception:
                pass

            avg_str_len = col_data.astype(str).apply(len).mean()

            if avg_str_len > 50:
                # Word cloud logic
                filtered_values = col_data.dropna().astype(str).tolist()
                filtered_values = [val for val in filtered_values if any(c.isalpha() for c in val)]
                text = ' '.join(filtered_values).strip()
                
                if text:
                    insights['wordcloud_text'] = text
                else:
                    # Fall back to bar chart
                    chart_df = pd.DataFrame({
                        col_name: val_counts_total.index.tolist(),
                        'Occurrences': val_counts_total.values
                    })
                    insights['charts'].append(('bar_chart', chart_df))
            else:
                chart_df = pd.DataFrame({
                    col_name: val_counts_total.index.tolist(),
                    'Occurrences': val_counts_total.values
                })
                insights['charts'].append(('bar_chart', chart_df))

            # Create value counts table
            percent_table = pd.DataFrame({
                "Value": val_counts_total.index,
                "Count": val_counts_total.values,
                "Percentage (%)": percent_total
            })
            percent_table = pd.concat([percent_table, pd.DataFrame([["Total", total_records, "100"]], columns=percent_table.columns)], ignore_index=True)
            percent_table["Percentage (%)"] = percent_table["Percentage (%)"].astype(float)
            insights['tables'].append(('value_counts', percent_table))

            if primary_keys and original_df is not None and col_name not in primary_keys:
                try:
                    # Create a DataFrame with just the columns we need
                    temp_df = original_df[primary_keys + [col_name]].copy()
                    # Drop rows where any of the primary keys or the column is null
                    temp_df = temp_df.dropna(subset=primary_keys + [col_name])
                    # Drop duplicates based on primary keys
                    temp_df = temp_df.drop_duplicates(subset=primary_keys)
                    
                    grouped_counts = temp_df[col_name].value_counts().head(top_n)
                    percent_keys = (grouped_counts / temp_df.shape[0] * 100).round(2)
                    
                    chart_df2 = pd.DataFrame({
                        col_name: grouped_counts.index.tolist(),
                        'Occurrences': grouped_counts.values
                    })
                    insights['charts'].append(('bar_chart_pk', chart_df2))
                    
                    percent_key_table = pd.DataFrame({
                        "Value": grouped_counts.index,
                        "Count": grouped_counts.values,
                        "Percentage (%)": percent_keys
                    })
                    percent_key_table = pd.concat([percent_key_table, pd.DataFrame([["Total", temp_df.shape[0], "100"]], columns=percent_key_table.columns)], ignore_index=True)
                    percent_key_table["Percentage (%)"] = percent_key_table["Percentage (%)"].astype(float)
                    insights['tables'].append(('value_counts_pk', percent_key_table))
                except Exception as e:
                    insights['error'] = f"Error in primary key processing: {str(e)}"
        else:
            # Numeric data - create histogram
            chart_df = pd.DataFrame({col_name: col_data})
            insights['charts'].append(('histogram', chart_df))

            # Add value counts table for numeric data too
            percent_table = pd.DataFrame({
                "Value": val_counts_total.index,
                "Count": val_counts_total.values,
                "Percentage (%)": percent_total
            })
            percent_table = pd.concat([percent_table, pd.DataFrame([["Total", total_records, "100"]], columns=percent_table.columns)], ignore_index=True)
            percent_table["Percentage (%)"] = percent_table["Percentage (%)"].astype(float)
            insights['tables'].append(('value_counts', percent_table))

        return insights
    except Exception as e:
        return {
            'column_name': col_name,
            'error': str(e)
        } 