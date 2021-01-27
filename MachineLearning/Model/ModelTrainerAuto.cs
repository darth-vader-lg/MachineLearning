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
   public class ModelTrainerAuto : IModelTrainer, IModelTrainerCycling
   {
      #region Properties
      /// <summary>
      /// Numero di folds di training
      /// </summary>
      public int NumFolds { get; set; } = 1;
      /// <summary>
      /// Numero massimo di tentativi di training del modello
      /// </summary>
      public int MaxTrainingCycles { get; set; } = int.MaxValue;
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
      ITransformer IModelTrainer.GetTrainedModel(ModelBase<MLContext> model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
      {
         var result = model.AutoTraining(data, int.MaxValue, out evaluationMetrics, NumFolds, cancellation);
         cancellation.ThrowIfCancellationRequested();
         return result;
      }
      #endregion
   }
}
