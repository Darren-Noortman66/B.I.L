# Brothers In Ladles (B.I.L) - Sage Price Update Tool

A web-based application designed to streamline the process of updating product prices between Sage and supplier files. This tool helps businesses efficiently manage price updates, cost adjustments, and maintain accurate pricing data.

![B.I.L Logo](static/img/BIL%20Logo.png)

## Features

- **File Upload Support**
  - Sage CSV file import
  - Supplier file import (supports both CSV and Excel formats)
  - Multi-sheet Excel file support

- **Smart Column Mapping**
  - Automatic header detection
  - Flexible column mapping for product codes and costs
  - Support for various file structures

- **Price Adjustment Options**
  - Supplier cost adjustments with percentage increases/decreases
  - Two-tier Price Exclusive calculations
  - Automatic handling of discontinued items
  - Preservation of Price Inclusive values

- **Data Processing**
  - Intelligent code matching between Sage and supplier files
  - Automatic handling of leading zeros and code formatting
  - Support for large datasets with automatic file splitting

- **User-Friendly Interface**
  - Step-by-step guided process
  - Clear instructions at each stage
  - Responsive design for all devices
  - Modern, clean interface with B.I.L branding

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Darren-Noortman66/B.I.L.git
   cd B.I.L
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at `http://localhost:5000`

## Usage Guide

1. **Upload Files**
   - Select your Sage CSV file
   - Select your supplier file (CSV or Excel)

2. **Select Excel Sheet** (if applicable)
   - If your supplier file is an Excel workbook with multiple sheets, select the appropriate sheet

3. **Choose Header Row**
   - Review the data preview
   - Select which row contains your column headers

4. **Map Columns**
   - Select which columns contain your product codes
   - Select which columns contain your cost information

5. **Configure Price Adjustments**
   - Set supplier cost adjustment percentage (optional)
   - Configure Price Exclusive calculations:
     - First percentage increase
     - Second percentage increase
   - Option to apply CaterBros increase

6. **Download Results**
   - Download the processed file(s)
   - For large datasets, files are automatically split into manageable chunks
   - Option to download all files as a ZIP archive

## File Requirements

### Sage File
- Must be in CSV format
- Must contain product codes and cost columns
- Price Inclusive column will be preserved (This is calculated by Sage after the update)

### Supplier File
- Can be CSV or Excel format
- Must contain product codes and updated costs
- Multiple sheets supported in Excel files

## Notes
- The tool automatically handles code matching between files
- Discontinued items are detected and marked appropriately
- Large files are automatically split into parts of 450 rows each
- All price calculations are performed with high precision

## Support

For support, please contact the development team or raise an issue in the GitHub repository.

## License

This project is proprietary software. All rights reserved.
