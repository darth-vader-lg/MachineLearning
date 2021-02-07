using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.Data.ClusteringPredictionTransformer<Microsoft.ML.Trainers.KMeansModelParameters>;
using TTrainer = Microsoft.ML.Trainers.KMeansTrainer;
using TOptions = Microsoft.ML.Trainers.KMeansTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe KMeansTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class KMeansTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal KMeansTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.Clustering.Trainers.KMeans(Options);
      #endregion
   }
}
