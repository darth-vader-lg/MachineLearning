using MachineLearning.Data;
using System;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Training con shuffle delle righe
   /// </summary>
   [Serializable]
   public class ModelTrainerShuffle : IModelTrainer, IModelTrainerCycling
   {
      #region Fields
      /// <summary>
      /// Seme per le operazioni random
      /// </summary>
      [NonSerialized]
      private int _trainingSeed;
      #endregion
      #region Properties
      /// <summary>
      /// Numero massimo di tentativi di training del modello
      /// </summary>
      public int MaxTrainingCycles { get; set; } = 1;
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
      CompositeModel IModelTrainer.GetTrainedModel(ModelBase model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
      {
         evaluationMetrics = null;
         var result = new CompositeModel(model) { model.GetPipes().Merged.Fit(data.CanShuffle ? model.ML.NET.Data.ShuffleRows(data, _trainingSeed++) : data) };
         result.Schema = data.Schema;
         cancellation.ThrowIfCancellationRequested();
         return result;
      }
      #endregion
   }
}
