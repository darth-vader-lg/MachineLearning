using MachineLearning.Data;
using Microsoft.ML;
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
      ITransformer IModelTrainer.GetTrainedModel(ModelBase<MLContext> model, IDataAccess data, out object evaluationMetrics, CancellationToken cancellation)
      {
         evaluationMetrics = null;
         var result = model.GetPipes().Merged.Fit(data);
         cancellation.ThrowIfCancellationRequested();
         return result;
      }
      #endregion
   }
}
