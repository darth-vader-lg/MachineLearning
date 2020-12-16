using Microsoft.ML;

namespace ML.Utilities.Models
{
   /// <summary>
   /// Interfaccia per i gestori di storage dei modelli
   /// </summary>
   public interface IModelStorage
   {
      #region Methods
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      ITransformer LoadModel(MLContext mlContext, out DataViewSchema schema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="model">Modello da salvare</param>
      /// <param name="mlContext">Contesto di machine learning</param>
      /// <param name="schema">Schema di input del modello</param>
      void SaveModel(MLContext mlContext, ITransformer model, DataViewSchema schema);
      #endregion
   }
}