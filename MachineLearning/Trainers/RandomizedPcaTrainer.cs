using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.PcaModelParameters;
using MSTrainer = Microsoft.ML.Trainers.RandomizedPcaTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe RandomizedPcaTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class RandomizedPcaTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal RandomizedPcaTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.AnomalyDetection.Trainers.RandomizedPca(Options);
      #endregion
   }
}
