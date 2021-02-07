using Microsoft.ML;

namespace MachineLearning.Trainers
{
   /// <summary>
   /// Catalogo di trainers di clustering
   /// </summary>
   public class CusteringTrainers : ContextProvider<MLContext>
   {
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      internal CusteringTrainers(IContextProvider<MLContext> context) : base(context) { }
      /// <summary>
      /// Restituisce un trainer di tipo KMeans
      /// </summary>
      /// <param name="options">Opzioni</param>
      /// <returns>Il trainer</returns>
      public KMeansTrainer AveragedPerceptron(Microsoft.ML.Trainers.KMeansTrainer.Options options = default) =>
         new KMeansTrainer(this, options);
      #endregion
   }
}
