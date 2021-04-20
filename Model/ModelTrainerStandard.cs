using MachineLearning.Data;
using System;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Trainer standard dei modelli
   /// </summary>
   [Serializable]
   public class ModelTrainerStandard : IModelTrainer
   {
      #region Methods
      /// <summary>
      /// Restituisce il modello effettuando il training
      /// </summary>
      /// <param name="model">Modello con cui effettuare il training</param>
      /// <param name="data">Dati di training</param>
      /// <param name="evaluationMetrics">Eventuali metriche di valutazione precalcolate</param>
      /// <param name="cancellation">Token di annullamento</param>
      /// <returns>Il modello appreso</returns>
      IDataTransformer IModelTrainer.GetTrainedModel(ModelBase model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
      {
         if (model is IModelTrainingStandard training)
            return training.StandardTraining(data, out evaluationMetrics, cancellation);
         evaluationMetrics = null;
         return null;
      }
      #endregion
   }
}
