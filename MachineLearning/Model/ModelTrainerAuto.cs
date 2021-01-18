using MachineLearning.Data;
using Microsoft.ML;
using System;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Trainer con ricerca automatica del miglior modello
   /// </summary>
   [Serializable]
   public class ModelTrainerAuto : IModelTrainer, IModelTrainerFolded
   {
      #region Properties
      /// <summary>
      /// Numero massimo di secondi di training
      /// </summary>
      public int MaxTimeInSeconds { get; set; } = 120;
      /// <summary>
      /// Numero di folds di training
      /// </summary>
      public int NumFolds { get; set; } = 1;
      #endregion
      #region Methods
      /// <summary>
      /// Restituisce il modello effettuando il training
      /// </summary>
      /// <param name="model">Modello con cui effettuare il training</param>
      /// <param name="data">Dati di training</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      ITransformer IModelTrainer.GetTrainedModel(ModelBase model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
      {
         var result = model.AutoTraining(data, MaxTimeInSeconds, out evaluationMetrics, NumFolds, cancellation);
         cancellation.ThrowIfCancellationRequested();
         return result;
      }
      #endregion
   }
}
