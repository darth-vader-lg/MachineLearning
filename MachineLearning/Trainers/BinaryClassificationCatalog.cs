using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di classificazione binaria
   /// </summary>
   public class BinaryClassificationCatalog : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal BinaryClassificationCatalog(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo AveragedPerceptron
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public AveragedPerceptronTrainer AveragedPerceptron(Microsoft.ML.Trainers.AveragedPerceptronTrainer.Options options = default) =>
         new AveragedPerceptronTrainer(this, options);
      #endregion
   }
}
