from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
import uuid
import pandas as pd
import numpy as np
import zipfile
from io import BytesIO
    
app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sage_file = request.files['sage_file']
        supplier_file = request.files['supplier_file']

        sage_filename = 'sage_' + str(uuid.uuid4()) + '.csv'
        supplier_filename = 'supplier_' + str(uuid.uuid4()) + os.path.splitext(supplier_file.filename)[1]

        sage_path = os.path.join(app.config['UPLOAD_FOLDER'], sage_filename)
        supplier_path = os.path.join(app.config['UPLOAD_FOLDER'], supplier_filename)

        sage_file.save(sage_path)
        supplier_file.save(supplier_path)

        session['sage_path'] = sage_path
        session['supplier_path'] = supplier_path

        # Check if supplier file is Excel and get sheet names
        if supplier_path.endswith(('.xlsx', '.xls')):
            try:
                excel_file = pd.ExcelFile(supplier_path)
                sheet_names = excel_file.sheet_names
                if len(sheet_names) > 1:
                    session['sheet_names'] = sheet_names
                    return redirect(url_for('select_sheet'))
                else:
                    # Only one sheet, proceed directly
                    df = pd.read_excel(supplier_path, header=None)
            except Exception as e:
                return f"Error reading supplier file: {e}"
        else:
            try:
                df = pd.read_csv(supplier_path, header=None)
            except Exception as e:
                return f"Error reading supplier file: {e}"

        sample_rows = df.head(5).astype(str).values.tolist()
        session['sample_rows'] = sample_rows

        return redirect(url_for('select_header'))

    return render_template('index.html')


@app.route('/select_sheet', methods=['GET', 'POST'])
def select_sheet():
    if request.method == 'POST':
        selected_sheet = request.form['selected_sheet']
        session['selected_sheet'] = selected_sheet
        
        supplier_path = session.get('supplier_path')
        try:
            df = pd.read_excel(supplier_path, sheet_name=selected_sheet, header=None)
            sample_rows = df.head(5).astype(str).values.tolist()
            session['sample_rows'] = sample_rows
            return redirect(url_for('select_header'))
        except Exception as e:
            return f"Error reading sheet {selected_sheet}: {e}"
    
    sheet_names = session.get('sheet_names', [])
    return render_template('select_sheet.html', sheet_names=sheet_names)


@app.route('/select_header', methods=['GET', 'POST'])
def select_header():
    if request.method == 'POST':
        selected_index = int(request.form['header_row'])
        session['header_index'] = selected_index

        supplier_path = session.get('supplier_path')
        selected_sheet = session.get('selected_sheet')
        
        if supplier_path.endswith('.csv'):
            df = pd.read_csv(supplier_path, header=selected_index)
        else:
            df = pd.read_excel(supplier_path, sheet_name=selected_sheet, header=selected_index)

        session['supplier_headers'] = list(df.columns)
        
        # Save the processed supplier data back to CSV for consistency
        temp_csv_path = supplier_path.replace('.xlsx', '.csv').replace('.xls', '.csv')
        df.to_csv(temp_csv_path, index=False)
        session['supplier_path'] = temp_csv_path

        return redirect(url_for('map_columns'))

    sample_rows = session.get('sample_rows', [])
    return render_template('select_header.html', sample_rows=sample_rows)


@app.route('/map_columns', methods=['GET', 'POST'])
def map_columns():
    if request.method == 'POST':
        session['code_col'] = request.form['code_col']
        session['cost_col'] = request.form['cost_col']
        return redirect(url_for('cost_adjustments'))

    headers = session.get('supplier_headers', [])
    return render_template('map_columns.html', headers=headers)


@app.route('/cost_adjustments', methods=['GET', 'POST'])
def cost_adjustments():
    if request.method == 'POST':
        # Get form data
        cost_increase_percentage = float(request.form.get('cost_increase', 0))
        price_exclusive_percentage_1 = float(request.form.get('price_exclusive_1', 0))
        price_exclusive_percentage_2 = float(request.form.get('price_exclusive_2', 0))
        add_caterbros_increase = 'add_caterbros_increase' in request.form
        
        # Store in session
        session['cost_increase_percentage'] = cost_increase_percentage
        session['price_exclusive_percentage_1'] = price_exclusive_percentage_1
        session['price_exclusive_percentage_2'] = price_exclusive_percentage_2
        session['add_caterbros_increase'] = add_caterbros_increase
        
        return redirect(url_for('process_files'))
    
    return render_template('cost_adjustments.html')


def split_dataframe(df, chunk_size=450):
    """Split a dataframe into chunks of specified size"""
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunks.append(df[i:i + chunk_size])
    return chunks


@app.route('/process_files')
def process_files():
    try:
        # Load files and session data
        sage_path = session.get('sage_path')
        supplier_path = session.get('supplier_path')
        code_col = session.get('code_col')
        cost_col = session.get('cost_col')
        cost_increase_percentage = session.get('cost_increase_percentage', 0)
        price_exclusive_percentage_1 = session.get('price_exclusive_percentage_1', 0)
        price_exclusive_percentage_2 = session.get('price_exclusive_percentage_2', 0)
        add_caterbros_increase = session.get('add_caterbros_increase', False)
        
        # Validate that all required data exists
        if not all([sage_path, supplier_path, code_col, cost_col]):
            return "Error: Missing required session data. Please start over."
        
        if not os.path.exists(sage_path):
            return f"Error: Sage file not found at {sage_path}"
        
        if not os.path.exists(supplier_path):
            return f"Error: Supplier file not found at {supplier_path}"
        
        # Load dataframes
        sage_df = pd.read_csv(sage_path)
        supplier_df = pd.read_csv(supplier_path)
        
        # Debug info
        print(f"Sage columns: {list(sage_df.columns)}")
        print(f"Supplier columns: {list(supplier_df.columns)}")
        print(f"Code column: {code_col}, Cost column: {cost_col}")
        
        # First identify any discontinued values in any column
        discontinued_terms = ['DISCON', 'DIS', 'DISCONTINUED', '"DISCON"', '"DIS"', '"DISCONTINUED"', "'DISCON'", "'DIS'", "'DISCONTINUED'"]
        discon_mask = supplier_df.apply(lambda x: x.astype(str).str.upper().str.strip('"\' ').isin(discontinued_terms)).any(axis=1)
        
        # Ensure the supplier cost column is numeric
        supplier_df[cost_col] = pd.to_numeric(supplier_df[cost_col], errors='coerce')
        
        # Convert identified discontinued rows to 0 while preserving other NaN values
        supplier_df.loc[discon_mask, cost_col] = 0
        
        # Apply cost increase to supplier data
        if cost_increase_percentage != 0:
            supplier_df[cost_col] = supplier_df[cost_col] * (1 + cost_increase_percentage / 100)
        
        # Normalize and clean the code columns for better matching
        def normalize_code(code):
            """Normalize codes for consistent matching"""
            if pd.isna(code):
                return None
            # Convert to string and strip whitespace
            code_str = str(code).strip()
            # Remove any leading zeros for numeric codes
            try:
                # If it's a numeric code, convert to int then back to string to remove leading zeros
                if code_str.replace('.', '').replace('-', '').isdigit():
                    code_str = str(int(float(code_str)))
            except:
                pass
            return code_str.upper()  # Make uppercase for case-insensitive matching
        
        # Normalize supplier codes
        supplier_df['normalized_code'] = supplier_df[code_col].apply(normalize_code)
        
        # Create a mapping dictionary from supplier data using normalized codes
        supplier_mapping = {}
        for idx, row in supplier_df.iterrows():
            norm_code = row['normalized_code']
            cost_value = row[cost_col]
            if norm_code is not None and pd.notna(cost_value):
                supplier_mapping[norm_code] = cost_value

        print(f"Created supplier mapping with {len(supplier_mapping)} entries")
        print(f"Sample supplier codes: {list(supplier_mapping.keys())[:10]}")
        
        # Find code and cost columns in Sage data (case-insensitive search)
        code_column_sage = None
        cost_column_sage = None
        
        # Look for code column
        for col in sage_df.columns:
            if 'code' in col.lower():
                code_column_sage = col
                break
        
        # Look for cost column
        for col in sage_df.columns:
            if 'cost' in col.lower():
                cost_column_sage = col
                break
        
        if not code_column_sage:
            return f"Error: Could not find code column in Sage file. Available columns: {list(sage_df.columns)}"
        
        if not cost_column_sage:
            return f"Error: Could not find cost column in Sage file. Available columns: {list(sage_df.columns)}"
        
        print(f"Using Sage columns - Code: {code_column_sage}, Cost: {cost_column_sage}")
        
        # Convert Sage cost column to numeric
        sage_df[cost_column_sage] = pd.to_numeric(sage_df[cost_column_sage], errors='coerce')
        
        # Normalize Sage codes for matching
        sage_df['normalized_code'] = sage_df[code_column_sage].apply(normalize_code)
        
        print(f"Sample Sage codes: {sage_df['normalized_code'].dropna().head(10).tolist()}")
        
        # Create a list to store only matched records
        matched_records = []
        matches_found = 0
        no_match_codes = []
        
        for idx, row in sage_df.iterrows():
            norm_code = row['normalized_code']
            if norm_code is not None and norm_code in supplier_mapping:
                new_cost = supplier_mapping[norm_code]
                if pd.notna(new_cost):
                    # Create a copy of the row and update the cost
                    updated_row = row.copy()
                    updated_row[cost_column_sage] = new_cost
                    
                    # Handle Price Exclusive column
                    price_exclusive_cols = [col for col in sage_df.columns if col == "Default Price List - Exclusive"]
                    if price_exclusive_cols:
                        price_exclusive_col = price_exclusive_cols[0]
                        print(f"Processing row with code {norm_code}, cost {new_cost}")
                        
                        # Set Price Exclusive to 0 for discontinued items first
                        if norm_code in supplier_mapping and supplier_mapping[norm_code] == 0:
                            print(f"Setting Price Exclusive to 0 for discontinued item {norm_code}")
                            updated_row[price_exclusive_col] = 0
                        else:
                            # Start with the new cost as the base price
                            updated_row[price_exclusive_col] = new_cost
                            
                            # Apply first percentage increase if specified
                            if price_exclusive_percentage_1 != 0:
                                updated_row[price_exclusive_col] = updated_row[price_exclusive_col] * (1 + price_exclusive_percentage_1 / 100)
                            
                            # Apply second percentage increase if specified
                            if price_exclusive_percentage_2 != 0:
                                updated_row[price_exclusive_col] = updated_row[price_exclusive_col] * (1 + price_exclusive_percentage_2 / 100)
                    
                    matched_records.append(updated_row)
                    matches_found += 1
            else:
                if norm_code is not None:
                    no_match_codes.append(norm_code)
        
        print(f"Found {matches_found} matching products")
        if no_match_codes:
            print(f"Sample codes with no matches: {no_match_codes[:10]}")
        
        if not matched_records:
            return "No matching records found between Sage and Supplier files. Please check your data and try again."
        
        # Convert matched records to DataFrame
        updated_sage = pd.DataFrame(matched_records)
        
        # Clean up temporary column
        updated_sage = updated_sage.drop('normalized_code', axis=1)
        
        # Ensure output directory exists
        output_dir = app.config['OUTPUT_FOLDER']
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Output directory confirmed: {output_dir}")
        except Exception as e:
            return f"Error creating output directory: {str(e)}"
        
        # Split the data into chunks if necessary
        output_files = []
        base_filename = f'updated_sage_{uuid.uuid4().hex[:8]}'
        
        if len(updated_sage) > 450:
            chunks = split_dataframe(updated_sage, 450)
            print(f"Splitting {len(updated_sage)} records into {len(chunks)} files")
            
            for i, chunk in enumerate(chunks, 1):
                chunk_filename = f'{base_filename}_part{i}.csv'
                chunk_path = os.path.abspath(os.path.join(output_dir, chunk_filename))
                
                try:
                    chunk.to_csv(chunk_path, index=False)
                    print(f"Created chunk file: {chunk_filename} with {len(chunk)} rows")
                    output_files.append({
                        'filename': chunk_filename,
                        'path': chunk_path,
                        'rows': len(chunk)
                    })
                except Exception as e:
                    return f"Error saving chunk file {chunk_filename}: {str(e)}"
        else:
            # Single file
            single_filename = f'{base_filename}.csv'
            single_path = os.path.abspath(os.path.join(output_dir, single_filename))
            
            try:
                updated_sage.to_csv(single_path, index=False)
                print(f"Created single file: {single_filename} with {len(updated_sage)} rows")
                output_files.append({
                    'filename': single_filename,
                    'path': single_path,
                    'rows': len(updated_sage)
                })
            except Exception as e:
                return f"Error saving file {single_filename}: {str(e)}"
        
        # Verify all files were created
        for file_info in output_files:
            if not os.path.exists(file_info['path']):
                return f"Error: Failed to create output file at {file_info['path']}"
            
            file_size = os.path.getsize(file_info['path'])
            if file_size == 0:
                return f"Error: Output file {file_info['filename']} was created but is empty"
            
            print(f"Verified file: {file_info['filename']} (Size: {file_size} bytes, Rows: {file_info['rows']})")
        
        # Store output information in session
        session['output_files'] = output_files
        session['matches_found'] = matches_found
        session['total_records'] = len(updated_sage)
        
        return redirect(url_for('download_result'))
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in process_files: {error_details}")
        return f"Error processing files: {str(e)}<br><br>Details:<br><pre>{error_details}</pre>"


@app.route('/download_result')
def download_result():
    output_files = session.get('output_files', [])
    matches_found = session.get('matches_found', 0)
    total_records = session.get('total_records', 0)
    
    if not output_files:
        return "Error: No output files found. Please try processing the files again."
    
    # Check that all files still exist
    for file_info in output_files:
        if not os.path.exists(file_info['path']):
            return f"Error: Output file {file_info['filename']} not found. Please try processing the files again."
    
    return render_template('download_result.html', 
                         output_files=output_files, 
                         matches_found=matches_found,
                         total_records=total_records)


@app.route('/download/<filename>')
def download_file(filename):
    # Find the file in the output_files list
    output_files = session.get('output_files', [])
    file_path = None
    
    for file_info in output_files:
        if file_info['filename'] == filename:
            file_path = file_info['path']
            break
    
    # Fallback: if not found in session, try to construct path
    if not file_path:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    
    # Verify the file exists before attempting to send it
    if not os.path.exists(file_path):
        return f"File not found: {filename}. Please try processing the files again.", 404
    
    try:
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return f"Error downloading file: {str(e)}", 500


@app.route('/download_all')
def download_all():
    """Download all output files as a ZIP archive"""
    output_files = session.get('output_files', [])
    
    if not output_files:
        return "Error: No output files found.", 404
    
    # Create a ZIP file in memory
    zip_buffer = BytesIO()
    
    try:
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_info in output_files:
                if os.path.exists(file_info['path']):
                    zip_file.write(file_info['path'], file_info['filename'])
                else:
                    return f"Error: File {file_info['filename']} not found.", 404
        
        zip_buffer.seek(0)
        
        return send_file(
            BytesIO(zip_buffer.read()),
            mimetype='application/zip',
            as_attachment=True,
            download_name='updated_sage_files.zip'
        )
    
    except Exception as e:
        return f"Error creating ZIP file: {str(e)}", 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
    
