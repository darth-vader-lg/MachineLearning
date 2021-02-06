using Microsoft.ML;
using System;
using MSModel = Microsoft.ML.Trainers.LinearBinaryModelParameters;
using MSTrainer = Microsoft.ML.Trainers.AveragedPerceptronTrainer;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Classe AveragedPerceptronTrainer con opzioni
   /// </summary>
   [Serializable]
   public sealed class AveragedPerceptronTrainer : TrainerBase<MSModel, MSTrainer, MSTrainer.Options>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="contextProvider">Provider di contesto di machine learning</param>
      internal AveragedPerceptronTrainer(IContextProvider<MLContext> contextProvider, MSTrainer.Options options = default) : base(contextProvider, options) { }
      /// <summary>
      /// Funzione di creazione del trainer
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <returns>Il trainer</returns>
      protected override MSTrainer CreateTrainer(MLContext context) => context.BinaryClassification.Trainers.AveragedPerceptron(Options);
      #endregion
   }
}
