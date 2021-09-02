using Microsoft.ML.Runtime;

namespace MachineLearning
{
   /// <summary>
   /// Interfaccia per i provider di contesto di machine learning
   /// </summary>
   /// <typeparam name="T">Tipo di contesto</typeparam>
   public interface IContextProvider<T> : IChannelProvider
   {
      #region Properties
      /// <summary>
      /// Contesto
      /// </summary>
      T Context { get; }
      #endregion
   }
}
