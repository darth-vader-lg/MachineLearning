using MachineLearning.Data;
using Microsoft.ML;

namespace MachineLearning.Model
{
   /// <summary>
   /// Interfaccia per i gestori di storage dei modelli
   /// </summary>
   public interface IModelStorage
   {
      #region Methods
      /// <summary>
      /// Eventuale path di importazione di un modello esterno (ONNX / TensorFlow, ecc...)
      /// </summary>
      public string ImportPath { get; }
      /// <summary>
      /// Funzione di caricamento modello
      /// </summary>
      /// <param name="context">Contesto di machine learning</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      /// <returns>Il modello</returns>
      ITransformer LoadModel(MLContext context, out DataSchema inputSchema);
      /// <summary>
      /// Funzione di salvataggio modello
      /// </summary>
      /// <param name="context">Contesto</param>
      /// <param name="model">Modello da salvare</param>
      /// <param name="inputSchema">Schema di input del modello</param>
      void SaveModel(MLContext context, ITransformer model, DataSchema inputSchema);
      #endregion
   }
}