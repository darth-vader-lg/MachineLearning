using Microsoft.ML;

namespace MachineLearning.Model
{
   /// <summary>
   /// Contenitore di pipes di training
   /// </summary>
   public class ModelPipes
   {
      #region Properties
      /// <summary>
      /// Pipe di input e featurizzazione
      /// </summary>
      public IEstimator<ITransformer> Input { get; set; }
      /// <summary>
      /// Pipe unita
      /// </summary>
      public IEstimator<ITransformer> Merged
      {
         get
         {
            var result = (IEstimator<ITransformer>)null;
            if (Input != null)
               result = result == null ? Input : result.Append(Input);
            if (Trainer != null)
               result = result == null ? Trainer : result.Append(Trainer);
            if (Output != null)
               result = result == null ? Output : result.Append(Output);
            return result;
         }
      }
      /// <summary>
      /// Pipe di output
      /// </summary>
      public IEstimator<ITransformer> Output { get; set; }
      /// <summary>
      /// Pipe di training
      /// </summary>
      public IEstimator<ITransformer> Trainer { get; set; }
      #endregion
   }
}
