// Paste this into Google Apps Script (Extensions > Apps Script)
// Then Deploy > New Deployment > Web App > Anyone can access > Deploy
// Copy the deployment URL and paste it into the Streamlit app

function doPost(e) {
  try {
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
    var data = JSON.parse(e.postData.contents);

    // If first row is empty, add headers
    if (sheet.getLastRow() === 0) {
      var headers = Object.keys(data);
      sheet.appendRow(headers);
    }

    // Append the data row
    var headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0];
    var row = headers.map(function(header) {
      return data[header] || "";
    });
    sheet.appendRow(row);

    return ContentService
      .createTextOutput(JSON.stringify({"status": "success"}))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (error) {
    return ContentService
      .createTextOutput(JSON.stringify({"status": "error", "message": error.toString()}))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  return ContentService
    .createTextOutput("AL Essentials Data Collection Endpoint - POST only")
    .setMimeType(ContentService.MimeType.TEXT);
}
