using Microsoft.ML;
using System;
using TModel = Microsoft.ML.Trainers.PcaModelParameters;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.PcaModelParameters>;
using TTrainer = Microsoft.ML.Trainers.RandomizedPcaTrainer;
using TOptions = Microsoft.ML.Trainers.RandomizedPcaTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe RandomizedPcaTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class RandomizedPcaTrainer : TrainerBase<TModel, TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal RandomizedPcaTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.AnomalyDetection.Trainers.RandomizedPca(Options);
      #endregion
   }
}
