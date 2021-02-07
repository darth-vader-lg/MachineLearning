using Microsoft.ML;
using System;
using TTransformer = Microsoft.ML.ISingleFeaturePredictionTransformer<Microsoft.ML.Trainers.LinearBinaryModelParameters>;
using TTrainer = Microsoft.ML.Trainers.AveragedPerceptronTrainer;
using TOptions = Microsoft.ML.Trainers.AveragedPerceptronTrainer.Options;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe AveragedPerceptronTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class AveragedPerceptronTrainer : TrainerBase<TTransformer, TTrainer, TOptions>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal AveragedPerceptronTrainer(IContextProvider<MLContext> contextProvider, TOptions options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override TTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.AveragedPerceptron(Options);
      #endregion
   }
}
