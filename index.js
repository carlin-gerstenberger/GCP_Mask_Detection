const webcamElement= document.getElementById('webcam');
let net;
let isPredicting = false;
function startPredicting(){
 isPredicting=true;
 app();
}
function stopPredicting(){
 isPredicting=false;
 app();
}
async function app(){
 console.log('Loading model..');
 net= await tf.automl.loadImageClassification('model.json');
 console.log('Successfully loaded model');
 
 const webcam = await tf.data.webcam(webcamElement);
 while(isPredicting){
 const img = await webcam.capture();
 const result = await net.classify(img);
 
 console.log(result);
 
 document.getElementById("mask").innerText=result['0']['label']+": "+Math.round(result['0']['prob']*100)+"%";
 document.getElementById("no_mask").innerText=result['1']['label']+": "+Math.round(result['1']['prob']*100)+"%";
 document.getElementById("incorrect_mask").innerText=result['2']['label']+": "+Math.round(result['2']['prob']*100)+"%";
img.dispose();
 
await tf.nextFrame();
 
 }
 
}