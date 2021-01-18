﻿using MachineLearning.Data;
using System;
using System.Threading;

namespace MachineLearning.Model
{
   /// <summary>
   /// Trainer con validazione incrociata
   /// </summary>
   [Serializable]
   public class ModelTrainerCrossValidation : IModelTrainer, IModelTrainerFolded
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
      /// Numero di folds di training
      /// </summary>
      public int NumFolds { get; set; } = 5;
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
         var result = model.CrossValidateTraining(data, out evaluationMetrics, NumFolds, null, _trainingSeed++);
         cancellation.ThrowIfCancellationRequested();
         return result;
      }
      #endregion
   }
}
