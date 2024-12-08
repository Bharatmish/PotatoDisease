import { useState, useEffect } from "react";
import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import React from "react";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";
import { Paper, CardActionArea, CardMedia, Grid, TableContainer, Table, TableBody, TableHead, TableRow, TableCell, Button, CircularProgress } from "@material-ui/core";
import cblogo from "./logoNavbar.png";
import image from "./background.jpg";
import { DropzoneArea } from 'material-ui-dropzone';
import { common } from '@material-ui/core/colors';
import Clear from '@material-ui/icons/Clear';

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    '&:hover': {
      backgroundColor: '#ffffff7a',
    },
  },
}))(Button);

const axios = require("axios").default;

const useStyles = makeStyles((theme) => ({
  grow: {
    flexGrow: 1,
  },
  clearButton: {
    width: "-webkit-fill-available",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#000000a6",
    fontSize: "20px",
    fontWeight: 900,
  },
  root: {
    maxWidth: 345,
    flexGrow: 1,
  },
  media: {
    height: 400,
  },
  paper: {
    padding: theme.spacing(2),
    margin: 'auto',
    maxWidth: 500,
  },
  gridContainer: {
    justifyContent: "center",
    padding: "4em 1em 0 1em",
  },
  mainContainer: {
    backgroundImage: `url(${image})`,
    backgroundRepeat: 'no-repeat',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    height: "93vh",
    marginTop: "8px",
  },
  imageCard: {
    margin: "auto",
    maxWidth: 400,
    height: 500,
    backgroundColor: 'transparent',
    boxShadow: '0px 9px 70px 0px rgb(0 0 0 / 30%) !important',
    borderRadius: '15px',
  },
  imageCardEmpty: {
    height: 'auto',
  },
  noImage: {
    margin: "auto",
    width: 400,
    height: "400 !important",
  },
  input: {
    display: 'none',
  },
  uploadIcon: {
    background: 'white',
  },
  tableContainer: {
    backgroundColor: 'transparent !important',
    boxShadow: 'none !important',
  },
  table: {
    backgroundColor: 'transparent !important',
  },
  tableHead: {
    backgroundColor: 'transparent !important',
  },
  tableRow: {
    backgroundColor: 'transparent !important',
  },
  tableCell: {
    fontSize: '22px',
    backgroundColor: 'transparent !important',
    borderColor: 'transparent !important',
    color: '#000000a6 !important',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableCell1: {
    fontSize: '14px',
    backgroundColor: 'transparent !important',
    borderColor: 'transparent !important',
    color: '#000000a6 !important',
    fontWeight: 'bolder',
    padding: '1px 24px 1px 16px',
  },
  tableBody: {
    backgroundColor: 'transparent !important',
  },
  text: {
    color: 'white !important',
    textAlign: 'center',
  },
  buttonGrid: {
    maxWidth: "416px",
    width: "100%",
  },
  detail: {
    backgroundColor: 'white',
    display: 'flex',
    justifyContent: 'center',
    flexDirection: 'column',
    alignItems: 'center',
  },
  appbar: {
    background: 'brown',
    boxShadow: 'none',
    color: 'white'
  },
  loader: {
    color: '#be6a77 !important',
  },

  // New styles for the summary section
  summaryContainer: {
    background: 'linear-gradient(135deg, #f5f5f5, #e0e0e0)',  // Soft gradient for background
    borderRadius: '15px',  // Rounded corners for the summary box
    padding: '25px',  // Increased padding for more breathing space
    marginTop: '30px', // More space from the prediction table
    boxShadow: '0px 12px 24px rgba(0, 0, 0, 0.1)',  // Larger shadow for more emphasis
    width: '100%',
    maxWidth: '850px', // Slightly wider max-width
    margin: 'auto',  // Center the summary box
    fontFamily: "'Roboto', sans-serif", // More modern font
    transition: 'transform 0.3s ease-in-out',  // Smooth hover effect
},

summaryTitle: {
    fontSize: '1.8rem',  // Larger title for better emphasis
    fontWeight: '600',  // Slightly lighter bold for a modern look
    marginBottom: '15px', // More space below the title
    color: '#2c3e50',  // Darker shade for a more sophisticated look
    textTransform: 'uppercase',  // Uppercase for an impactful look
    letterSpacing: '1px',  // Adds spacing between letters for style
    textShadow: '2px 2px 4px rgba(0, 0, 0, 0.2)', // Soft text shadow for depth
},

summaryText: {
    fontSize: '1.1rem',  // Slightly larger text for better readability
    lineHeight: '1.8',  // Increased line height for improved text flow
    color: '#34495e',  // Darker but softer text color
    whiteSpace: 'pre-line',  // Preserve new lines in the summary text
    wordWrap: 'break-word',  // Ensure long words break
    letterSpacing: '0.5px',  // Add spacing for a cleaner look
    textAlign: 'justify',  // Justify the text for a neat appearance
    marginBottom: '20px', // Space at the bottom of the text
    backgroundColor: '#f9f9f9',  // Slight background color to highlight the text
    padding: '10px',  // Padding to make the text feel more spacious
    borderRadius: '8px',  // Rounded corners for the text block
    boxShadow: 'inset 0px 4px 8px rgba(0, 0, 0, 0.1)', // Subtle inset shadow
},

}));

const ImageUpload = () => {
  const classes = useStyles();
  const [selectedFile, setSelectedFile] = useState();
  const [preview, setPreview] = useState();
  const [data, setData] = useState();
  const [isLoading, setIsloading] = useState(false);
  const [showSummary, setShowSummary] = useState(false);  // New state for toggling summary visibility
  let confidence = 0;

  const sendFile = async () => {
    if (selectedFile) {
      let formData = new FormData();
      formData.append("file", selectedFile);
      setIsloading(true); // Show loading spinner

      try {
        let res = await axios({
          method: "post",
          url: process.env.REACT_APP_API_URL, // Ensure this points to your backend
          data: formData,
        });

        if (res.status === 200) {
          setData(res.data); // Save the API response
        }
      } catch (error) {
        console.error("Error uploading file:", error);
        alert("Failed to process the image. Please try again.");
      } finally {
        setIsloading(false); // Hide loading spinner
      }
    }
  };

  const clearData = () => {
    setData(null);
    setSelectedFile(null);
    setPreview(null);
    setShowSummary(false);  // Reset the summary visibility when clearing data
  };

  useEffect(() => {
    if (!selectedFile) {
      setPreview(undefined);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  }, [selectedFile]);

  useEffect(() => {
    if (!preview) {
      return;
    }
    setIsloading(true);
    sendFile();
  }, [preview]);

  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      setSelectedFile(undefined);
      setData(undefined);
      return;
    }
    setSelectedFile(files[0]);
    setData(undefined);
  };

  if (data) {
    confidence = (parseFloat(data.confidence) * 100).toFixed(2);
  }

  // Function to toggle summary visibility
  const toggleSummary = () => {
    setShowSummary(!showSummary);
  };

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography className={classes.title} variant="h6" noWrap>
            Potato-disease Classification
          </Typography>
          <div className={classes.grow} />
          <Avatar src={cblogo}></Avatar>
        </Toolbar>
      </AppBar>
      <Container maxWidth={false} className={classes.mainContainer} disableGutters={true}>
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <Card className={`${classes.imageCard} ${!selectedFile ? classes.imageCardEmpty : ''}`}>
              {selectedFile && (
                <CardActionArea>
                  <CardMedia
                    className={classes.media}
                    image={preview}
                    component="image"
                    title="Potato Leaf"
                  />
                </CardActionArea>
              )}
              {!selectedFile && (
                <CardContent className={classes.content}>
                  <DropzoneArea
                    acceptedFiles={['image/*']}
                    dropzoneText={"Drag & Drop an image of a potato plant leaf to Process"}
                    onChange={onSelectFile}
                  />
                </CardContent>
              )}
              
              {data && (
                <CardContent className={classes.detail}>
                  <TableContainer component={Paper} className={classes.tableContainer}>
                    <Table className={classes.table} size="small" aria-label="simple table">
                      <TableHead className={classes.tableHead}>
                        <TableRow className={classes.tableRow}>
                          <TableCell className={classes.tableCell1}>Label:</TableCell>
                          <TableCell align="right" className={classes.tableCell1}>Confidence:</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody className={classes.tableBody}>
                        <TableRow className={classes.tableRow}>
                          <TableCell component="th" scope="row" className={classes.tableCell}>
                            {data.class}
                          </TableCell>
                          <TableCell align="right" className={classes.tableCell}>{confidence}%</TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              )}
              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress color="secondary" className={classes.loader} />
                  <Typography className={classes.title} variant="h6" noWrap>
                    Processing
                  </Typography>
                </CardContent>
              )}
            </Card>
          </Grid>

          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="large" onClick={clearData} startIcon={<Clear fontSize="large" />}>
                Clear
              </ColorButton>
            </Grid>
          )}

          {data && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton variant="contained" className={classes.clearButton} color="primary" component="span" size="large" onClick={toggleSummary}>
                {showSummary ? "Hide Summary" : "Show Summary"}
              </ColorButton>
            </Grid>
          )}

          {showSummary && data && (
            <Grid item xs={12} className={classes.buttonGrid}>
              <CardContent className={classes.summaryContainer}>
                <Typography className={classes.summaryTitle}>Summary:</Typography>
                <Typography className={classes.summaryText}>
                  {data.summary}
                </Typography>
              </CardContent>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );


  
};


export default ImageUpload;
