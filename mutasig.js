
mutasig={ classNames: [], x_train: {}, x_test: [], y_train: [], y_test: [], model: {}, epochLogs: [] }

mutasig.loadData = async (url) => {
    //mutasig.model = await tf.loadLayersModel(url_model);

    await fetch( url ).then( (response) => response.json() ).then( (data) => {
        mutasig.x_train = tf.tensor(data.x_train);
        mutasig.x_test = tf.tensor(data.x_test)
        
        mutasig.y_train = tf.tensor(data.y_train);
        mutasig.y_test = tf.tensor(data.y_test);
        
        mutasig.classNames = data.classNames;
    });
}

mutasig.loadModel = () => {
    mutasig.model = tf.sequential({
     layers: [
       tf.layers.dense({inputShape: [96], units: 128, activation: 'relu'}),
       tf.layers.dense({ units: 256, activation: 'relu'}),
       tf.layers.dense({ units: mutasig.classNames.length }),
     ]
    });
    
    mutasig.model.compile({
      optimizer: 'adam',
      loss: 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });
}

mutasig.train = (fitCallbacks) => {
  const BATCH_SIZE = 32;
  
  return mutasig.model.fit( mutasig.x_train, mutasig.y_train, {
        batchSize: BATCH_SIZE,
        validationData: [ mutasig.x_test, mutasig.y_test ],
        epochs: 50,
        shuffle: true,
        callbacks: fitCallbacks
      });
}

mutasig.train_visualization = async () => {
    const callbacks = {
        onEpochEnd: function (epoch, log) {
          const surface = {
            name: 'Training Visualization',
            tab: 'Training'
          };
          
          const options = {
            xLabel: 'Epoch',
            yLabel: 'Value',
            yAxisDomain: [0, 1],
            seriesColors: ['teal', 'tomato']
          }; // Prep the data

          mutasig.epochLogs.push(log);
          
          const acc = mutasig.epochLogs.map((log, i) => ({
            x: i,
            y: log.acc
          }));
          
          const valAcc = mutasig.epochLogs.map((log, i) => ({
            x: i,
            y: log.val_acc
          }));
          
          const data = {
            values: [acc, valAcc],
            // Custom names for the series
            series: ['Accuracy', 'Validation Accuracy'] // render the chart

          };
          tfvis.render.linechart(surface, data, options);
        }   
  };
  
  return mutasig.train(callbacks);
}

mutasig.doPrediction = () => {
    const labels = mutasig.y_test;
    const preds = mutasig.model.predict(mutasig.x_test).argMax([-1]);
    return [preds, labels];
}

mutasig.showAccuracy = async () => {
    const [preds, labels] = mutasig.doPrediction();
    console.log(labels.shape);
    console.log(preds.shape);
    // Use preds.data() to extract the values from tensor
    
    const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
    
    const container = {
      name: 'Accuracy',
      tab: 'Evaluation'
    };
    
    tfvis.show.perClassAccuracy(container, classAccuracy, mutasig.classNames);
}

mutasig.showConfusion = async () => {
    const [preds, labels] = mutasig.doPrediction();
    
    const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
    
    const container = {
      name: 'Confusion Matrix',
      tab: 'Evaluation'
    };
    
    tfvis.render.confusionMatrix(container, {
      values: confusionMatrix,
      tickLabels: mutasig.classNames
    });
}

